"""Single-vector retrieval index over the merged corpus, torch GPU brute-force (no faiss).
Loads every block vector from a paradigm's merged manifest ({doc,corpus,path} -> {doc}.npy
[n_blocks,dim]) into one [N,dim] fp16 GPU tensor, with a parallel meta list (doc, corpus,
block_row). search(qvec, k) -> top-k [(meta, score)]. block_row indexes into the doc's blocks
(corpus.body_figures / body_pages / chunks order). jina_v4_multi (figure, dir-of-per-fig) uses
the separate MaxSim path -- not this loader.
"""
import json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch


class Index:
    def __init__(self, X, meta, device):
        self.X = X.to(device)            # [N, dim] fp16, L2-normalized
        self.meta = meta                 # list[(doc, corpus, block_row)]
        self.device = device

    def search(self, qvec, k=5):
        q = torch.as_tensor(np.asarray(qvec), dtype=torch.float16, device=self.device).reshape(1, -1)
        q = q / (q.float().norm() + 1e-9).half()
        scores = (self.X @ q.T).squeeze(1)               # cosine (both L2-normalized)
        kk = min(k, scores.numel())
        top = torch.topk(scores, kk)
        return [(self.meta[i], float(top.values[j])) for j, i in enumerate(top.indices.tolist())]


_FB = {"nf": None}   # NPZ_FALLBACK: missing per-doc npys resolve from the derived _all.npz

# Manifests store ABSOLUTE paths under /path/to/work -- node-local storage on NODE. Any
# job that lands on another berkeleynlp node (lorax, horton) finds none of them, every row falls to
# the fallback, and the index silently degenerates. This is the same defect corpus.py fixed with
# _norm and eval_rpv2_multi.py fixed with _normimg; the index layer never got it.
_PATH_ROOTS = ("/path/to/work", "/path/to/work")

def _norm_path(p):
    if os.path.exists(p):
        return p
    for a in _PATH_ROOTS:
        if p.startswith(a + "/"):
            for b in _PATH_ROOTS:
                if b == a:
                    continue
                q = b + p[len(a):]
                if os.path.exists(q):
                    return q
    return p


def _read(it):
    try:
        a = np.load(_norm_path(it["path"]))
    except FileNotFoundError:
        nf = _FB.get("nf")
        if nf is not None and it.get("doc") in getattr(nf, "files", []):
            a = nf[it["doc"]]
        else:
            _FB["skipped"] = _FB.get("skipped", 0) + 1
            _FB["total"] = _FB.get("total", 0)
            if _FB["skipped"] % 500 == 1:
                print("NPZ_FALLBACK: %d rows have no vectors anywhere -> zero-stub (never retrieved)" % _FB["skipped"], flush=True)
            import numpy as _n
            a = _n.zeros((0, 4096), dtype="float16")   # no rows at all, not zero rows
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return it["doc"], it["corpus"], a.astype(np.float16)


def _derive_npz(manifest_path):
    # merged_manifests/p3__qwen3_vl_embedding_8b_602k.jsonl -> corpus_cs/embeddings/p3/qwen3_vl_embedding_8b_602k_all.npz
    base = os.path.basename(manifest_path)
    if base.endswith(".jsonl"):
        base = base[:-6]
    # A "__full" manifest spans the 99k corpus and its vectors live in <sub>_full.npz. Stripping the
    # suffix sent it to <sub>_all.npz, the 67,843-key CS-only pack, whose keys mostly do not match
    # the merged manifest's doc ids -- so 99,001 of 99,135 rows fell through to the zero-stub branch
    # below and search_figure returned the same five documents for every query. That is what voided
    # every txtfig "_figfix" run.
    _want_full = "__full" in base
    base = base.replace("__benchmerged", "").replace("__full", "").replace("__withnew2", "")
    sub = base.replace("__", "/")
    root = os.environ.get("VLDR_ROOT", "/path/to/vldr")
    if _want_full:
        cand = os.path.join(root, "corpus_cs", "embeddings", sub + "_full.npz")
        if os.path.exists(cand):
            return cand
    return os.path.join(root, "corpus_cs", "embeddings", sub + "_all.npz")


def load_npz(npz_path, device="cuda"):
    """Load a packed {doc}.npy-per-member .npz (the CS-corpus merged embedding, arxiv-keyed) into
    one [N,dim] fp16 tensor. Replaces the deleted per-doc loose .npy (packed Jul28 to fix inodes)."""
    corp = os.environ.get("VLDR_CORPUS", "cs")
    nf = np.load(npz_path)                              # NpzFile over raw .npy members; keys drop the .npy
    arrs, meta = [], []
    for k in nf.files:
        a = nf[k]
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.shape[0] == 0:
            continue
        a = a.astype(np.float16)
        doc = k[:-4] if k.endswith(".npy") else k
        arrs.append(a)
        meta.extend((doc, corp, br) for br in range(a.shape[0]))
    X = np.concatenate(arrs, axis=0)
    Xt = torch.from_numpy(X)
    Xt = Xt / (Xt.float().norm(dim=1, keepdim=True) + 1e-9).half()
    print("index NPZ %s: %d vectors, dim=%d, docs=%d" % (
        npz_path.split("/")[-1], Xt.shape[0], Xt.shape[1], len(nf.files)), flush=True)
    return Index(Xt, meta, device)



def load_olddir(manifest_path, device="cuda", workers=32):
    """OLD corpus: each doc's vectors live in a per-doc DIR of per-page npy (doc_XXX/{0,1,..}.npy).
    The manifest lists doc_XXX.npy (packed away Jul28); the real data is the sibling directory.
    Loads the docs that still have a dir (covers the 1305 train golds). VLDR_OLDDIR=1 to use."""
    import glob
    items = [json.loads(l) for l in open(manifest_path) if l.strip()]
    def _read(it):
        pth = it.get("path", "")
        d = pth[:-4] if pth.endswith(".npy") else pth
        if not os.path.isdir(d):
            return None
        fs = glob.glob(d + "/*.npy")
        if not fs:
            return None
        fs.sort(key=lambda f: int(os.path.basename(f)[:-4]) if os.path.basename(f)[:-4].isdigit() else (1 << 30))
        rows = []
        for f in fs:
            try:
                rows.append(np.load(f).reshape(-1))
            except Exception:
                pass
        if not rows:
            return None
        return it["doc"], it.get("corpus", "old"), np.stack(rows).astype(np.float16)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        loaded = [x for x in ex.map(_read, items) if x]
    # Refuse to hand back an index that is mostly zero stubs. A run on such an index looks healthy
    # in the log, burns the full API budget, and returns the same handful of documents for every
    # query -- it must fail loudly here instead.
    _sk = _FB.get("skipped", 0)
    if _sk and _sk > 0.01 * len(items):
        raise RuntimeError(
            "index %s: %d of %d manifest rows have no vectors (%.1f%%). The derived npz does not "
            "cover this manifest -- fix the npz before running." % (
                os.path.basename(manifest_path), _sk, len(items), 100.0 * _sk / len(items)))
    dim = next(a.shape[1] for _, _, a in loaded if a.shape[0] > 0)
    N = sum(a.shape[0] for _, _, a in loaded)
    X = np.empty((N, dim), dtype=np.float16); meta = []; off = 0
    for doc, corp, a in loaded:
        n = a.shape[0]
        if n == 0:
            continue
        X[off:off + n] = a
        meta.extend((doc, corp, br) for br in range(n)); off += n
    Xt = torch.from_numpy(X[:off]); Xt = Xt / (Xt.float().norm(dim=1, keepdim=True) + 1e-9).half()
    print("index OLDDIR %s: %d vectors, dim=%d, docs=%d" % (manifest_path.split("/")[-1], off, dim, len(loaded)), flush=True)
    return Index(Xt, meta, device)


def load_combined(manifest_path, device="cuda", workers=32):
    """FAITHFUL original corpus = EXACTLY the manifest's docs (old+new1). Pulls each doc's vectors from
    the OLD per-doc-dir (hash docs) or the CS _all.npz (arxiv new1 docs). EXCLUDES new2 (not in manifest);
    skips docs whose loose npy were deleted Jul28 (unrecoverable). Matches the new-harness rerun corpus."""
    import glob
    items = [json.loads(l) for l in open(manifest_path) if l.strip()]
    root = os.environ.get("VLDR_ROOT", "/path/to/vldr")
    a2d = json.load(open(os.path.join(root, "meta", "arxiv_to_doc.json")))
    d2a = {v: k for k, v in a2d.items()}
    npz = _derive_npz(manifest_path)
    nf = np.load(npz) if os.path.exists(npz) else None
    cs_keys = set(nf.files) if nf is not None else set()
    # __OLD_NPZ_PREFERRED__
    _pdir = os.path.dirname(items[0]["path"]) if items and items[0].get("path") else ""
    _oldnpz = _pdir + ".npz"
    _onf = np.load(_oldnpz) if _pdir and os.path.exists(_oldnpz) else None
    _onf_keys = set(_onf.files) if _onf is not None else set()
    _onf_loaded = []
    if _onf is not None:
        for _it in items:
            _doc = _it["doc"]
            if _doc in _onf_keys:
                _a = _onf[_doc]
                if _a.ndim == 1: _a = _a.reshape(1, -1)
                _onf_loaded.append((_doc, _it.get("corpus", "old"), _a.astype(np.float16)))
    onf_docs = set(x[0] for x in _onf_loaded)
    # pass 1 (threaded): old per-doc-dir vectors
    def _old(it):
        if it["doc"] in onf_docs: return None
        pth = it.get("path", ""); d = pth[:-4] if pth.endswith(".npy") else pth
        if not os.path.isdir(d):
            if pth.endswith(".npy") and os.path.isfile(pth):   # bench per-doc .npy = [n_pages, dim]
                try:
                    a = np.load(pth)
                    if a.ndim == 1: a = a.reshape(1, -1)
                    return it["doc"], it.get("corpus", "bench"), a.astype(np.float16)
                except Exception:
                    return None
            return None
        fs = glob.glob(d + "/*.npy")
        if not fs: return None
        fs.sort(key=lambda f: int(os.path.basename(f)[:-4]) if os.path.basename(f)[:-4].isdigit() else (1 << 30))
        rows = []
        for f in fs:
            try: rows.append(np.load(f).reshape(-1))
            except Exception: pass
        if not rows: return None
        return it["doc"], it.get("corpus", "old"), np.stack(rows).astype(np.float16)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        got = list(ex.map(_old, items))
    loaded = [x for x in got if x] + _onf_loaded
    have = set(x[0] for x in loaded)
    # pass 2 (single-thread, NpzFile not thread-safe): arxiv docs from cs npz
    n_cs = 0
    for it in items:
        doc = it["doc"]
        if doc in have: continue
        ax = doc if not doc.startswith("doc_") else d2a.get(doc)
        if ax and ax in cs_keys:
            a = nf[ax]
            if a.ndim == 1: a = a.reshape(1, -1)
            loaded.append((doc, it.get("corpus", "new"), a.astype(np.float16))); n_cs += 1
    dim = next(a.shape[1] for _, _, a in loaded if a.shape[0] > 0)
    N = sum(a.shape[0] for _, _, a in loaded)
    X = np.empty((N, dim), dtype=np.float16); meta = []; off = 0
    for doc, corp, a in loaded:
        n = a.shape[0]
        if n == 0: continue
        X[off:off + n] = a; meta.extend((doc, corp, br) for br in range(n)); off += n
    Xt = torch.from_numpy(X[:off]); Xt = Xt / (Xt.float().norm(dim=1, keepdim=True) + 1e-9).half()
    print("index COMBINED(manifest): %d/%d docs (old-dir=%d + cs-npz=%d), %d vectors, dim=%d" % (
        len(loaded), len(items), len(have), n_cs, off, dim), flush=True)
    return Index(Xt, meta, device)


def load(manifest_path, device="cuda", workers=32):
    try:
        _np_fb = _derive_npz(manifest_path)
        import os as _o
        if _o.path.exists(_np_fb):
            _FB["nf"] = np.load(_np_fb)
        else:
            _FB["nf"] = None
    except Exception:
        _FB["nf"] = None
    _pk = os.environ.get("IDX_PACKED")
    if _pk and os.path.exists(_pk + ".X.npy"):
        _X = np.load(_pk + ".X.npy")
        _meta = [tuple(m) for m in json.load(open(_pk + ".meta.json"))]
        print("index PACKED: %d vectors" % len(_meta), flush=True)
        return Index(torch.from_numpy(_X), _meta, device)
    if os.environ.get("VLDR_COMBINED"):
        return load_combined(manifest_path, device, workers)
    if os.environ.get("VLDR_OLDDIR"):
        return load_olddir(manifest_path, device, workers)
    # CS-corpus retrieval: loose per-doc .npy were packed into _all.npz (Jul28 inode fix). When
    # VLDR_USE_NPZ=1 and the packed file exists, load from it instead of the (now-dead) manifest paths.
    if os.environ.get("VLDR_USE_NPZ"):
        npz = _derive_npz(manifest_path)
        if os.path.exists(npz):
            return load_npz(npz, device)
    items = [json.loads(l) for l in open(manifest_path) if l.strip()]
    # parallel file reads: 60k tiny .npy on VAST is open-latency bound -> threads overlap the waits
    with ThreadPoolExecutor(max_workers=workers) as ex:
        loaded = list(ex.map(_read, items))
    dim = next(a.shape[1] for _, _, a in loaded if a.shape[0] > 0)
    N = sum(a.shape[0] for _, _, a in loaded)
    X = np.empty((N, dim), dtype=np.float16)
    meta = []
    off = 0
    for doc, corp, a in loaded:
        n = a.shape[0]
        if n == 0:
            continue
        X[off:off + n] = a
        meta.extend((doc, corp, br) for br in range(n))
        off += n
    X = X[:off]
    Xt = torch.from_numpy(X)
    Xt = Xt / (Xt.float().norm(dim=1, keepdim=True) + 1e-9).half()   # ensure L2-normalized
    # A degenerate row would normalize to NaN, and torch.topk ranks NaN ABOVE every real score, so
    # one bad row is enough to own the top-k of every query. Zero them so they can never be returned.
    Xt = torch.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    _sk = _FB.get("skipped", 0)
    _cov = 100.0 * (len(items) - _sk) / max(len(items), 1)
    print("index %s: %d vectors, dim=%d | %d/%d docs have vectors (%.1f%% coverage), %d skipped"
          % (manifest_path.split("/")[-1], off, dim, len(items) - _sk, len(items), _cov, _sk), flush=True)
    if _cov < 80.0:
        raise RuntimeError("index %s covers only %.1f%% of its manifest -- the derived npz is wrong "
                           "for this manifest. Refusing to run." % (os.path.basename(manifest_path), _cov))
    return Index(Xt, meta, device)
