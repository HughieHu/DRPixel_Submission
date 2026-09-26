"""Unified accessor over the MERGED corpus (old 17k + new CS 42.8k). Resolves each doc's
content_list / images.zip / origin.pdf (old: raw/Arxiv_*, doc_<hash> via doc_to_arxiv; new:
corpus_cs/ocr, doc==stem) and serves the content a retrieved block needs:
  chunks(doc)        -> list[str]   (256-word chunks, reading-order text -- matches build_text)
  body_figures(doc)  -> list[{hash,page_idx}]  (type==image on body pages -- matches build_figures)
  body_pages(doc)    -> list[int]   (page_idx order -- matches build_pages)
  figure_crop(doc,h) -> jpeg bytes  (from images.zip)
  page_image(doc,pi) -> png bytes   (fitz render @ dpi)
Block ORDER mirrors the embedding scripts, so embedding row i == block i here.
"""
import os, json, glob, io, zipfile, functools
import fitz  # PyMuPDF

ROOT = os.environ.get("VLDR_ROOT", "/path/to/vldr")   # Berkeley: /path/to/vldr
NEW_OCR = ROOT + "/corpus_cs/ocr"
OLD_RAW = ROOT + "/raw"
META = ROOT + "/meta"

# ---- block helpers (copied from embedding_build/common.py + build_text.py for self-containment) ----
import re
_SKIP_TEXT = {"page_number", "footer", "page_footnote"}

def _load(cl):
    return json.load(open(cl))

def _body_pages(d):
    pages = sorted({x.get("page_idx", 0) for x in d})
    cut = None
    for x in d:
        t = (x.get("text") or "").strip().lower().rstrip(":")
        if t and (x.get("text_level") or x.get("type") == "title") and t in ("references", "reference", "bibliography"):
            pi = x.get("page_idx"); cut = pi if cut is None else min(cut, pi)
    return [p for p in pages if cut is None or p <= cut]

def _paper_text(d):
    parts = []
    for x in d:
        t = x.get("type")
        if t in _SKIP_TEXT:
            continue
        if t == "table":
            cap = " ".join(x.get("table_caption") or []).strip()
            if cap: parts.append(cap)
            b = (x.get("table_body") or "").strip()
            if b: parts.append(b)
        elif t == "image":
            cap = " ".join(x.get("image_caption") or []).strip()
            if cap: parts.append(cap)
        else:
            s = (x.get("text") or "").strip()
            if s: parts.append(s)
    return "\n\n".join(parts)

def _clean(text):
    text = re.sub(r'^---\n.*?\n---\n?', '', text, flags=re.DOTALL)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def _chunk(text, n=256):
    w = text.split()
    return [' '.join(w[i:i + n]) for i in range(0, len(w), n) if w[i:i + n]]

# ---- doc -> file paths (built once, cached to JSON: globbing 60k dirs on VAST is slow) ----
_DOC_MAP_CACHE = os.environ.get("VLDR_DOCMAP") or (ROOT + "/experiment/doc_map.json")
# VLDR_DOCMAP: the stored paths reflect whichever node wrote the file; _norm() below maps
# them onto the running ROOT, so no per-node copy or sed-rewrite of doc_map.json is needed.

BENCH_ROOT = ROOT + "/corpus_new3/bench_corpora"

def _bench_doc_map():
    m = {}
    for cl in glob.glob(BENCH_ROOT + "/irpapers/ocr/misc/*/*_content_list.json"):
        doc = os.path.basename(os.path.dirname(cl))
        m[doc] = {"corpus": "irpapers", "cl": cl,
                  "zip": os.path.join(os.path.dirname(cl), "images.zip"),
                  "pdf": BENCH_ROOT + "/irpapers/papers/%s.pdf" % doc}
    for cl in glob.glob(BENCH_ROOT + "/m3sciqa/ocr/*/*/*_content_list.json"):
        doc = os.path.basename(os.path.dirname(cl))
        m[doc] = {"corpus": "m3sciqa", "cl": cl,
                  "zip": os.path.join(os.path.dirname(cl), "images.zip"),
                  "pdf": BENCH_ROOT + "/m3sciqa/pdfs/%s.pdf" % doc}
    for cl in glob.glob(BENCH_ROOT + "/airqa/ocr/misc/*/*_content_list.json"):
        doc = os.path.basename(os.path.dirname(cl))
        m[doc] = {"corpus": "airqa", "cl": cl,
                  "zip": os.path.join(os.path.dirname(cl), "images.zip"),
                  "pdf": os.path.join(os.path.dirname(cl), doc + "_origin.pdf")}
    return m

@functools.lru_cache(maxsize=1)
def _doc_map():
    if os.path.exists(_DOC_MAP_CACHE):
        m = json.load(open(_DOC_MAP_CACHE))
    else:
        m = _build_doc_map()
        try:
            json.dump(m, open(_DOC_MAP_CACHE, "w"))
        except Exception:
            pass
    if os.environ.get("VLDR_BENCH"):
        m = dict(m); m.update(_bench_doc_map())
    return m

def _build_doc_map():
    m = {}
    # new corpus: doc == stem
    for d in glob.glob(NEW_OCR + "/*/*/"):
        stem = os.path.basename(d.rstrip("/"))
        cl = "%s%s_content_list.json" % (d, stem)
        if os.path.exists(cl):
            m[stem] = {"corpus": "new", "cl": cl, "zip": d + "images.zip",
                       "pdf": "%s%s_origin.pdf" % (d, stem)}
    # old corpus: doc == doc_<hash>; map via arxiv_to_doc
    a2d = json.load(open(META + "/arxiv_to_doc.json"))
    raw_by_arxiv = {}
    for d in glob.glob(OLD_RAW + "/Arxiv_*/*/*/"):
        aid = os.path.basename(d.rstrip("/"))
        raw_by_arxiv[aid] = d
    for aid, doc in a2d.items():
        d = raw_by_arxiv.get(aid)
        if not d or doc in m:
            continue
        cl = glob.glob(d + "*_content_list.json")
        pdf = glob.glob(d + "*_origin.pdf")
        if cl:
            m[doc] = {"corpus": "old", "cl": cl[0], "zip": d + "images.zip",
                      "pdf": pdf[0] if pdf else None}
    return m

# ---- node-independent paths -------------------------------------------------
# doc_map.json stores ABSOLUTE paths, and which absolute path it stores depends on
# which node wrote it (NODE /data, the js pool /net/NODE/data, or scratch).
# A stored path is therefore rewritten onto whatever ROOT *this* process is using,
# so ONE doc_map.json is correct on every node. Longest prefix first, and the match
# must end on a path separator -- ".../VL_DR" must never swallow ".../VL_DR_net",
# which is exactly the bug that produced 22-31% search errors on the js pool.
_KNOWN_ROOTS = tuple(sorted((
    "/path/to/vldr",
    "/path/to/vldr",
    "/path/to/vldr",
    "/path/to/work/VL_DR",
), key=len, reverse=True))


def _norm(p):
    if not isinstance(p, str):
        return p
    for pre in _KNOWN_ROOTS:
        if p.startswith(pre) and (len(p) == len(pre) or p[len(pre)] == "/"):
            return ROOT + p[len(pre):]
    return p


# A missing PDF is served as a BLANK page (below), which is invisible to the agent and
# to the grader -- the failure mode that silently poisoned 15 rollouts once. Count them
# so a preflight/health check can see it, and let VLDR_STRICT_PAGES=1 turn it into a crash.
_MISSING = {"n": 0, "docs": set()}


def missing_report():
    """(count, sample docs) of page fetches that fell back to a blank image."""
    return _MISSING["n"], sorted(_MISSING["docs"])[:20]


def info(doc):
    d = _doc_map().get(doc)
    return {k: _norm(v) for k, v in d.items()} if d else None

@functools.lru_cache(maxsize=4096)
def _cl_cache(doc):
    i = info(doc)
    return _load(i["cl"]) if i else []

def chunks(doc, n=256):
    return _chunk(_clean(_paper_text(_cl_cache(doc))), n)

def body_figures(doc):
    d = _cl_cache(doc); bset = set(_body_pages(d))
    return [{"hash": os.path.basename(x["img_path"]), "page_idx": x.get("page_idx")}
            for x in d if x.get("type") == "image" and x.get("img_path") and x.get("page_idx") in bset]

def body_pages(doc):
    return _body_pages(_cl_cache(doc))

def figure_crop(doc, h):
    i = info(doc)
    if not i or not os.path.exists(i["zip"]):
        return None
    with zipfile.ZipFile(i["zip"]) as zf:
        names = set(zf.namelist())
        name = h if h in names else next((n for n in names if os.path.basename(n) == h), None)
        return zf.read(name) if name else None

@functools.lru_cache(maxsize=64)
def _pdf_bytes(path):
    """Cache whole PDFs: page_image re-read the file for every page, which costs
    nothing on node-local /data but is ~8x the NFS traffic when ROOT is /net/<node>."""
    return open(path, "rb").read()


def page_image(doc, page_idx, dpi=150):
    i = info(doc)
    if not i or not i["pdf"] or not os.path.exists(i["pdf"]):
        _MISSING["n"] += 1; _MISSING["docs"].add(doc)
        if os.environ.get("VLDR_STRICT_PAGES"):
            raise FileNotFoundError("page_image: no PDF for %s -> %r" % (doc, i and i.get("pdf")))
        import io as _io
        from PIL import Image as _Img
        _b = _io.BytesIO(); _Img.new("RGB", (1024, 1400), "white").save(_b, "PNG")
        return _b.getvalue()  # BLANK page fallback (PDF not staged, e.g. new2 distractor)
    with fitz.open(stream=_pdf_bytes(i["pdf"]), filetype="pdf") as pdf:
        if 0 <= page_idx < len(pdf):
            page = pdf[page_idx]; zoom = dpi / 72.0
            _mp = int(os.environ.get("IMG_MAX_PIXELS", "0"))   # cap page-image resolution (602112 ~= 588 qwen tok)
            if _mp:
                import math as _math
                _px = (page.rect.width * zoom) * (page.rect.height * zoom)
                if _px > _mp: zoom *= _math.sqrt(_mp / _px)
            return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)).tobytes("png")
    return None
