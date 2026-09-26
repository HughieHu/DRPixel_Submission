"""Retriever = index (merged-corpus vectors) + query encoder + corpus content resolution.
search(query, k=5) -> list of hits, each with the content the agent reads:
  text   : {doc, corpus, score, chunk_idx, text}
  figure : {doc, corpus, score, hash, page_idx, image}   (image = jpeg bytes of the crop)
  page   : {doc, corpus, score, page_idx, image}          (image = png bytes of the rendered page)
block_row from the index maps into corpus.chunks/body_figures/body_pages (same order as the
embedding scripts). jina_v4_multi (MaxSim) is handled by a separate retriever (figure expand).
"""
import os, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import index as _index
import encoders as _encoders
import corpus as _corpus

import os
MANIFEST_DIR = os.environ.get("VLDR_ROOT", "/path/to/vldr") + "/merged_manifests"
_IDXL = os.environ.get("IDX_LOCAL")
def _redir(path):
    if _IDXL and "/embeddings/p3/" in path:
        return _IDXL.rstrip("/") + "/" + path.split("/embeddings/p3/", 1)[1]
    return path

# paradigm -> (manifest file, encoder kind, modality)
PARADIGMS = {
    "text":   ("text__qwen3_embedding_8b.jsonl",     "text",       "text"),
    "fig_q8": ("p4__qwen3_vl_embedding_8b.jsonl",     "qwen_vl_8b", "figure"),
    # 99,135-doc merged figure index (old+new+new2+airqa/m3sciqa/irpapers). The plain fig_q8
    # manifest covers only 60,197 docs (old+new) -- every txtfig number before 2026-08-26 was
    # retrieved from that smaller pool. Kept side by side so old runs stay reproducible.
    "fig_q8_full": ("p4__qwen3_vl_embedding_8b__benchmerged__full.jsonl", "qwen_vl_8b", "figure"),
    "fig_q2": ("p4__qwen3_vl_embedding_2b.jsonl",     "qwen_vl_2b", "figure"),
    "fig_js": ("p4__jina_v4_single.jsonl",            "jina_single", "figure"),
    "fig_jm": ("p4__jina_v4_multi.jsonl",             "jina_multi", "figure"),   # MaxSim -- separate path
    "pg_q8":  ("p3__qwen3_vl_embedding_8b_602k.jsonl", "qwen_vl_8b", "page"),
    "pg_q2":  ("p3__qwen3_vl_embedding_2b_602k.jsonl", "qwen_vl_2b", "page"),
    "pg_js":  ("p3__jina_v4_single.jsonl",            "jina_single", "page"),
    "pg_jm":  ("p3__jina_v4_multi.jsonl",             "jina_multi", "page"),    # MaxSim -- separate path
    # text at other chunk lengths (returned passage re-chunked to match)
    "text_c1024": ("text__qwen3_embedding_8b_chunk1024.jsonl", "text", "text"),
    "text_c4096": ("text__qwen3_embedding_8b_chunk4096.jsonl", "text", "text"),
    # page at other Qwen3-VL resolutions (same query encoder; page image render unchanged)
    "pg_r512":  ("p3__qwen3_vl_embedding_8b_res512.jsonl",  "qwen_vl_8b", "page"),
    "pg_r1280": ("p3__qwen3_vl_embedding_8b_res1280.jsonl", "qwen_vl_8b", "page"),
}
# word-chunk size per text paradigm (so search() returns the matching-length passage)
TEXT_CHUNK = {"text": 256, "text_c1024": 1024, "text_c4096": 4096}


class Retriever:
    def __init__(self, paradigm, device="cuda"):
        if paradigm not in PARADIGMS:
            raise ValueError("unknown paradigm %r" % paradigm)
        manifest, enc_kind, modality = PARADIGMS[paradigm]
        self.paradigm = paradigm
        self.modality = modality
        self.chunk_size = TEXT_CHUNK.get(paradigm, 256)
        self.enc = _encoders.build_encoder(enc_kind, device)
        self.idx = _index.load(os.path.join(MANIFEST_DIR, manifest), device)

    def search(self, query, k=5):
        qv = self.enc.encode_query(query)
        out = []
        for (doc, corp, br), score in self.idx.search(qv, k):
            r = {"doc": doc, "corpus": corp, "score": round(score, 4)}
            if self.modality == "text":
                chs = _corpus.chunks(doc, self.chunk_size)
                r["chunk_idx"] = br
                r["text"] = chs[br] if br < len(chs) else ""
            elif self.modality == "figure":
                figs = _corpus.body_figures(doc)
                if br < len(figs):
                    r["hash"] = figs[br]["hash"]
                    r["page_idx"] = figs[br]["page_idx"]
                    r["image"] = _corpus.figure_crop(doc, r["hash"])
            elif self.modality == "page":
                pgs = _corpus.body_pages(doc)
                if br < len(pgs):
                    r["page_idx"] = pgs[br]
                    r["image"] = _corpus.page_image(doc, pgs[br])
            out.append(r)
        return out


class MaxSimRetriever:
    """jina_v4 MULTI-vector retrieval via ColBERT-style MaxSim (late interaction).
    FIGURE: it['path'] = dir of {block_idx}.npy [n_tok,128]. PAGE: {doc}.npz keys p{idx} -> [n_tok,128].
    Scores each block by sum_qtok max_btok <q,b>. modality = figure | page.
    MULTI-GPU: the token tensor is SHARDED across `nshard` GPUs (cuda:0..) -- pages are ~230GB and
    don't fit on one card; shards are contiguous block ranges balanced by token count, scored in
    parallel and concatenated (global block order preserved)."""
    def __init__(self, device="cuda", modality="figure", manifest="p4__jina_v4_multi.jsonl",
                 workers=32, nshard=1):
        self.device = device
        self.modality = modality
        self.enc = _encoders.JinaEncoder("cuda:0" if nshard > 1 else device)
        self.nshard = max(1, nshard)

        def _readdoc(it):
            path = _redir(it["path"]); blks = []
            if path.endswith(".npz"):
                try:
                    z = np.load(path)
                    for kk in z.files:
                        if kk.startswith("p") and kk[1:].isdigit():
                            blks.append((int(kk[1:]), z[kk].astype(np.float16)))
                except Exception: pass
            else:
                try:
                    for f in os.listdir(path):
                        if f.endswith(".npy") and f[:-4].isdigit():
                            blks.append((int(f[:-4]), np.load(os.path.join(path, f)).astype(np.float16)))
                except FileNotFoundError: pass
            blks.sort()
            return it["doc"], it["corpus"], blks

        items = [json.loads(l) for l in open(os.path.join(MANIFEST_DIR, manifest)) if l.strip()]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            loaded = list(ex.map(_readdoc, items))
        blocks, self.meta = [], []
        for doc, corp, blks in loaded:
            for blk_idx, arr in blks:
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                blocks.append(arr); self.meta.append((doc, corp, blk_idx))
        del loaded
        self.n_blk = len(blocks)
        # token-balanced contiguous block-range boundaries across nshard GPUs
        sizes = [b.shape[0] for b in blocks]; total = sum(sizes)
        bounds, cum, tgt = [0], 0, total / self.nshard
        for i, sz in enumerate(sizes):
            cum += sz
            if cum >= tgt and len(bounds) < self.nshard:
                bounds.append(i + 1); tgt += total / self.nshard
        bounds.append(self.n_blk)
        self.shards = []   # (dev, tok[Ttok_s,128], tokblk_local[Ttok_s], b0, n_local_blk)
        for s in range(self.nshard):
            b0, b1 = bounds[s], bounds[s + 1]
            sub = blocks[b0:b1]
            tok_s = np.concatenate(sub, 0)
            blk_s = np.concatenate([np.full(a.shape[0], i, np.int64) for i, a in enumerate(sub)])
            dev = "cuda:%d" % s if self.nshard > 1 else device
            T = torch.from_numpy(tok_s)
            T = (T / (T.float().norm(dim=1, keepdim=True) + 1e-9).half()).to(dev)
            self.shards.append((dev, T, torch.from_numpy(blk_s).to(dev), b0, b1 - b0))
            for i in range(b0, b1):
                blocks[i] = None                      # free this shard's CPU source
            del sub, tok_s, blk_s
            print("  shard %d -> %s: %d blocks, %d tokens" % (s, dev, b1 - b0, T.shape[0]), flush=True)
        print("maxsim index (%s, %d-GPU): %d blocks total" % (modality, self.nshard, self.n_blk), flush=True)

    def search(self, query, k=5):
        Qnp = self.enc.encode_query_multi(query)                          # [nq,128] numpy
        parts = []
        for dev, tok, tokblk, b0, nloc in self.shards:
            Q = torch.from_numpy(Qnp).to(dev).half()                      # [nq,128]
            S = Q @ tok.T                                                 # [nq, Ttok_shard]
            nq = S.shape[0]
            per = torch.full((nq, nloc), -1e4, device=dev, dtype=S.dtype)
            per.scatter_reduce_(1, tokblk.unsqueeze(0).expand(nq, -1), S, reduce="amax", include_self=True)
            parts.append(per.sum(0).float().cpu())                        # [nloc]
        maxsim = torch.cat(parts)                                         # [n_blk] global block order
        top = torch.topk(maxsim, min(k, self.n_blk))
        out = []
        for j, g in enumerate(top.indices.tolist()):
            doc, corp, blk_idx = self.meta[g]
            r = {"doc": doc, "corpus": corp, "score": round(float(top.values[j]), 4)}
            if self.modality == "figure":
                figs = _corpus.body_figures(doc)
                if blk_idx < len(figs):
                    r["hash"] = figs[blk_idx]["hash"]; r["page_idx"] = figs[blk_idx]["page_idx"]
                    r["image"] = _corpus.figure_crop(doc, r["hash"])
            else:
                pgs = _corpus.body_pages(doc)
                if blk_idx < len(pgs):
                    r["page_idx"] = pgs[blk_idx]; r["image"] = _corpus.page_image(doc, pgs[blk_idx])
            out.append(r)
        return out


class VisitTool:
    """visit(doc_id): return the FULL cleaned markdown/text of ONE paper (body + table/figure captions).
    `query` = a 'doc ...' id shown in search results. Lets the agent read a whole paper instead of chunks.
    Not a retriever (no GPU). Text capped at 250k chars (~60k tok) to guard pathological papers."""
    modality = "visit"
    MAXCHARS = 250000
    def __init__(self, *a, **k):
        pass
    def search(self, query, k=1):
        doc = (query or "").strip().replace("doc ", "").strip()
        try:
            cl = _corpus._cl_cache(doc)
        except Exception:
            cl = None
        if not cl:
            return [{"doc": doc, "corpus": "?",
                     "text": "(visit failed: %r is not a known doc id -- pass an exact 'doc ...' id "
                             "exactly as shown in a search result)" % doc}]
        md = _corpus._clean(_corpus._paper_text(cl))[: self.MAXCHARS]
        return [{"doc": doc, "corpus": (_corpus.info(doc) or {}).get("corpus", "?"), "text": md}]


class VisitPagesTool:
    """visit(doc_id) for the PAGE modality: return ALL whole-page images of ONE paper (not text).
    `query` = a 'doc ...' id from a search result. Cap 30 pages to guard pathological long papers."""
    modality = "visit_pages"
    # default 30 (gpt/gemini runs); set VISIT_MAX_PAGES=12 for qwen so the visit step fits its context/image cap
    MAXPAGES = int(os.environ.get("VISIT_MAX_PAGES", "30"))
    def __init__(self, *a, **k):
        pass
    def search(self, query, k=1):
        doc = (query or "").strip().replace("doc ", "").strip()
        try:
            pages = _corpus.body_pages(doc)[: self.MAXPAGES]
        except Exception:
            pages = []
        if not pages:
            return [{"doc": doc, "corpus": "?",
                     "text": "(visit failed: %r is not a known doc id -- pass an exact 'doc ...' id "
                             "exactly as shown in a search result)" % doc}]
        corp = (_corpus.info(doc) or {}).get("corpus", "?")
        out = []
        for pi in pages:
            try:
                out.append({"doc": doc, "corpus": corp, "page_idx": pi, "image": _corpus.page_image(doc, pi)})
            except Exception:
                pass
        return out


def make(paradigm, device="cuda"):
    # VLDR_IDX_DEVICE override: run.py passes args.device="cuda" explicitly, defeating any
    # default-arg env lookup. Intercept HERE so both the query encoder and the 8GB page index
    # can be kept off a GPU that sglang (or another user) already filled.
    _envdev = __import__("os").environ.get("VLDR_IDX_DEVICE")
    if _envdev:
        device = _envdev
    """Factory: MaxSim retriever for fig_jm / pg_jm, single-vec Retriever otherwise.
    pg_jm shards across ALL visible GPUs (page MaxSim index ~230GB needs >=2 cards)."""
    if paradigm == "visit":
        return VisitTool()
    if paradigm == "visit_pages":
        return VisitPagesTool()
    if paradigm == "fig_jm":
        return MaxSimRetriever(device, modality="figure", manifest="p4__jina_v4_multi.jsonl", nshard=1)
    if paradigm == "pg_jm":
        ng = max(1, torch.cuda.device_count())
        return MaxSimRetriever(device, modality="page", manifest="p3__jina_v4_multi.jsonl", nshard=ng)
    return Retriever(paradigm, device)
