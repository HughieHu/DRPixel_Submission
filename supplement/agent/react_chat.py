import json, os, re, time, threading, collections

import agent_model as gem

HARNESS_VERSION = "chat-v1"

TOOL_DESCS = {
    "search_text":   'search_text: retrieve the top-5 most relevant TEXT passages from the paper corpus.',
    "search_figure": 'search_figure: retrieve the top-5 most relevant FIGURE images from the corpus (you will SEE them).',
    "search_page":   'search_page: retrieve the top-3 most relevant whole-PAGE images from the corpus (you will SEE them).',
    "visit":         'visit: open ONE whole paper and read ALL of it. Pass a doc id EXACTLY as shown in a tool result.',
}
K_BY_TOOL = {"search_text": 5, "search_figure": 5, "search_page": 3, "visit": 1}

for _t in list(K_BY_TOOL):
    _v = os.environ.get("K_" + _t.upper())
    if _v:
        K_BY_TOOL[_t] = int(_v)
for _t, _k in K_BY_TOOL.items():
    if _t in TOOL_DESCS:
        TOOL_DESCS[_t] = re.sub(r"top-\d+", "top-%d" % _k, TOOL_DESCS[_t])

OBS_DESCS = {
    "search_text":   '  - search_text -> <passage id=T# doc=DOC_ID>the passage text</passage>',
    "search_figure": '  - search_figure -> <figure id=F# doc=DOC_ID> followed by the FIGURE IMAGE (you can see it)',
    "search_page":   '  - search_page -> <page id=P# doc=DOC_ID page=N> followed by the PAGE IMAGE (you can see it)',
    "visit":         '  - visit -> the full paper text, or every page image of that paper',
}

SYSTEM = """You are a research assistant who answers questions about scientific papers through iterative reasoning and research.

## Process
- Show your reasoning inside <think></think> tags.
- Call a tool with exactly:  <call_tool name="TOOL_NAME">your query</call_tool>
- When you have enough evidence, give the final answer with exactly:  <answer>your answer</answer>
{BREVITY}
- You may alternate between thinking and tool calls several times.

## STOP after the tag  (this is the most important rule)
Each turn you emit AT MOST ONE <call_tool> or ONE <answer>, and then you STOP WRITING.
Do NOT write a <tool_output> block yourself -- WE run the tool and send you the real results.
Anything you write after the closing </call_tool> is discarded, and any tool output you invent
is a hallucination: the documents, page numbers and numbers in it are not real.

Your turn looks like exactly this and nothing more:

    <think>What I know so far, and what I still need.</think>
    <call_tool name="TOOL_NAME">my query</call_tool>

Your <think> stays visible to you for the rest of the session, so record in it any concrete
value you read off an image -- that is your working memory.

## Tools (use ONLY these)
{tools}

## Tool Output
After each tool call the results come back to you in a <tool_output> block:
{obs}
Results from EVERY tool call stay visible to you for the rest of the session -- you can re-read an
image you retrieved earlier.

## Before you answer -- check yourself
You may only answer when you can point to the SPECIFIC page/passage id you read the answer off
(e.g. "P2_1"). Ask yourself:
  - Is the exact value/claim the question asks for actually VISIBLE in one of the results I received?
  - If NO -> do NOT answer. Search again with a different query.
  - "The retrieved page is about the right topic" is NOT enough. "I remember this paper" is NOT enough.
Answering from memory when the evidence is not in front of you is the single worst failure mode here,
because the answer will look confident and be about the wrong paper.

## CRITICAL ANSWERING RULES
- Answer ONLY from evidence you actually retrieved this session. Do NOT answer from prior knowledge.
- When the answer involves a specific number, score, magnitude, threshold, or a named
  model/method/dataset, you MUST retrieve it and state the SPECIFIC value(s). A vague or directional
  answer ("generally better", "tends to improve", "competitive", "around X") when a precise value
  exists is treated as WRONG.
- NEVER guess or fabricate numbers, model names, method names, arxiv ids or document ids. If after
  searching you still cannot find the specific value the question asks for, say so explicitly --
  do NOT invent one.
- You start this session having retrieved NOTHING. The ONLY evidence that exists is what comes back
  inside a real <tool_output> block that WE send you. If you have not yet received a <tool_output>,
  you have no evidence and you MUST NOT answer -- search first.
- Recalling the paper from memory is not evidence. If your answer would be the same without looking
  at the retrieved pages, you have not done the task.
- For yes/no, comparison, trend, or regime questions: state the verdict/direction AND the supporting
  numbers behind it.

Illustrative GOOD vs BAD answers (fabricated examples, NOT the real question):
- Q "Does pruning hurt accuracy on the small model?"
  BAD : "Yes, pruning generally lowers accuracy somewhat."            (vague -> WRONG)
  GOOD: "Yes -- accuracy drops from 79.3% to 74.1% at 50% sparsity (Table 3), a 5.2-point fall."
- Q "Which retriever wins on the multilingual benchmark, and by how much?"
  BAD : "A single general retriever is competitive across languages."  (names nothing -> WRONG)
  GOOD: "Retriever-Z leads, +1.8 nDCG@10 over the per-language baseline (Table 4)."

Rules: you may call tools at most {budget} times in total; after that you MUST answer.
Issue focused queries. Read the returned passages/images carefully."""

SYSTEM_NATIVE_PROCESS = """You are a research assistant who answers questions about scientific papers through iterative reasoning and research.

## Process
- Before every tool call, write your reasoning as VISIBLE text inside <think></think> tags, then make
  the call. That <think> block is your working memory: it stays in the conversation and you can re-read
  it later, so record in it every concrete value you have read off an image and what you still need.
- Call one of your tools when you need evidence. We run it and send you the real results.
- Results come back through the tool channel: text passages as <passage id=...>, and figure/page
  IMAGES that you can actually SEE and read numbers off.
- Every result you receive stays visible to you for the rest of the session -- you can re-read an
  image you retrieved earlier.
- When you have enough evidence, stop calling tools and write your final answer as plain text.
- You may call tools at most {budget} times in total; after that you MUST answer.
"""

IMG_HINT = ("\n\nIMPORTANT -- you also have an IMAGE tool ({imgtools}). Some questions can ONLY be answered by "
            "looking at a figure, chart, plot, or table image: the exact number / curve shape / comparison is NOT "
            "in the body text. Whenever a question refers to a figure/table/chart, or asks for a value you cannot "
            "find in retrieved text, you MUST use the image tool and READ the actual image.")

_SEARCH_LOCK = threading.Lock()

_CALL_RE = re.compile(r'<call_tool\s+name\s*=\s*"([^"]+)"\s*>(.*?)</call_tool>', re.S | re.I)

ANSWER_MAX_CHARS = int(os.environ.get("ANSWER_MAX_CHARS", "0"))
ANSWER_BRIEF = os.environ.get("ANSWER_BRIEF", "0") == "1"

_BREVITY = ("- Keep the final answer SHORT: one or two sentences that state the verdict and the\n"
            "  specific value(s) that support it. Do not restate your reasoning, do not list\n"
            "  alternative candidate answers." if ANSWER_BRIEF else "")
SYSTEM = SYSTEM.replace("{BREVITY}", _BREVITY)

_OPTQ = """
----- HOW TO WRITE A GOOD PAGE SEARCH QUERY (read before every search) -----
A search query is NOT a question -- it is a bag of the exact terms you expect to appear ON the target page.
Tables and dense numeric pages are found by the VALUES and LABELS printed on them, not by a description.
QUOTE the exact values/category labels you expect ("21,000", "pe-negative"); name the Table by number
("Table II Table III"); add the page number if hinted ("page 9"); include the metric column names. Broad
query first, then a narrower quoted one.

search_page (top-3 whole-page images you SEE):
  GOOD: "21,000" "315,000" fixed bugs unfixed bugs tokens RepairAgent Fig. 9 cost per bug page 9
  GOOD: "pe-negative" baseline Table II Table III Table IV GPT-3.5 GPT-4o vulnerable samples percentage
  GOOD: "Video Alchemist" background reference subject references page 15 ablation FID
  BAD : table with the results                               (no value, no table number -> lands on wrong page)
---------------------------------------------------------------------------
"""
if os.environ.get("PROMPT_OPTQ") == "1":
    SYSTEM = SYSTEM + _OPTQ
    SYSTEM_NATIVE_PROCESS = SYSTEM_NATIVE_PROCESS + _OPTQ

_NLQ = """
----- HOW TO WRITE A GOOD PAGE SEARCH QUERY (read before every search) -----
Write your search query as ONE fluent natural-language sentence (about 20-30 words) that describes the SINGLE page you need. Name the paper's method/topic, the specific quantities / entities / metrics / comparison at stake, and the exhibit (table or figure) if the value lives there. Write natural language -- NOT a keyword bag, NOT quoted fragments.
  GOOD: Find the page with the table comparing CFGen and MultiVI for multimodal RNA and ATAC generation, reporting the MMD metric.
  GOOD: Find the page in the rolling-shutter correction paper that presents the homography formulation for camera motion, specifically the pure-rotation reduction.
  GOOD: Find the page with the figure or table comparing refusal rates across religious personas for GPT-4o and Llama3-70b.
  BAD : table with the results   (no method, no metric, no exhibit -> lands on the wrong page)
---------------------------------------------------------------------------
"""
if os.environ.get("PROMPT_NLQ") == "1":
    SYSTEM = SYSTEM + _NLQ
    SYSTEM_NATIVE_PROCESS = SYSTEM_NATIVE_PROCESS + _NLQ

if os.environ.get("AGENT_MODEL","").startswith("gpt-"):
    SYSTEM = SYSTEM.replace(
        "You are a research assistant who answers questions about scientific papers through iterative reasoning and research.",
        "You answer questions about scientific papers through iterative reasoning and research.", 1)

    SYSTEM_NATIVE_PROCESS = SYSTEM_NATIVE_PROCESS.replace(
        "You are a research assistant who answers questions about scientific papers through iterative reasoning and research.",
        "You answer questions about scientific papers through iterative reasoning and research.", 1)

_ANS_RE  = re.compile(r'<answer>(.*?)</answer>', re.S | re.I)

def _close_tags(txt):
    t = txt or ""
    if t.count("<call_tool") > t.count("</call_tool>"):
        t += "</call_tool>"
    if t.count("</think>") > t.count("<think>"):
        t = "<think>" + t
    if t.count("<think>") > t.count("</think>"):
        t += "</think>"
    return t

_THINK_RE = re.compile(r'<think>(.*?)</think>', re.S | re.I)

def _split_think(txt):
    m = _THINK_RE.search(txt or "")
    return m.group(1).strip() if m else ""

def _parse(txt):
    txt = _close_tags(txt)
    mc = _CALL_RE.search(txt)
    ma = _ANS_RE.search(txt)
    if mc and (not ma or mc.start() < ma.start()):
        return "call", mc.group(1).strip(), mc.group(2).strip(), txt[:mc.end()]
    if ma:
        return "answer", ma.group(1).strip(), None, txt[:ma.end()]
    if "<answer>" in txt:
        _a = txt.split("<answer>", 1)[1].strip()
        if ANSWER_MAX_CHARS and len(_a) > ANSWER_MAX_CHARS:
            _a = _a[:ANSWER_MAX_CHARS]
        return "answer", _a, None, txt
    return "none", txt.strip(), None, txt

_QWRAP_RE = re.compile(r'^\s*query\s*=\s*', re.I)

def _clean_query(q):
    if not q:
        return q
    t = q.strip()
    if t.startswith("{"):
        try:
            d = json.loads(t)
            if isinstance(d, dict) and d.get("query"):
                return str(d["query"]).strip()
        except Exception:
            pass
    m = _QWRAP_RE.match(t)
    if m:
        t = t[m.end():].strip()
        if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
            t = t[1:-1]
        elif t.startswith(("\"", "'")):
            t = t[1:]
        return t.strip()
    return q

def _hit_record(tool, h, hid):
    r = {"id": hid, "doc": h["doc"], "corpus": h.get("corpus"), "score": h.get("score")}
    for k in ("page_idx", "hash", "chunk_idx"):
        if k in h:
            r[k] = h[k]
    if tool in ("search_text",) or (tool == "visit" and h.get("image") is None):
        r["text"] = (h.get("text") or "")[:700]
    r["has_image"] = h.get("image") is not None
    return r

def _format_obs(tool, hits, step):
    parts = [gem.text_part("<tool_output>")]
    recs = []
    for i, h in enumerate(hits):
        if tool == "search_text":
            hid = "T%d_%d" % (step, i)
            parts.append(gem.text_part('<passage id=%s doc=%s>%s</passage>'
                                       % (hid, h["doc"], (h.get("text") or "")[:700])))
        elif tool == "search_figure":
            hid = "F%d_%d" % (step, i)
            parts.append(gem.text_part('<figure id=%s doc=%s>' % (hid, h["doc"])))
            if h.get("image"):
                parts.append(gem.img_bytes(h["image"], "image/jpeg"))
        elif tool == "search_page":
            hid = "P%d_%d" % (step, i)
            parts.append(gem.text_part('<page id=%s doc=%s page=%s>' % (hid, h["doc"], h.get("page_idx"))))
            if h.get("image"):
                parts.append(gem.img_bytes(h["image"], "image/png"))
        else:
            if h.get("image") is not None:
                hid = "V%d_%d" % (step, i)
                parts.append(gem.text_part('<page id=%s doc=%s page=%s>' % (hid, h["doc"], h.get("page_idx"))))
                parts.append(gem.img_bytes(h["image"], "image/png"))
            else:
                hid = "V%d_%d" % (step, i)
                parts.append(gem.text_part('<paper doc=%s>%s</paper>' % (h["doc"], h.get("text") or "")))
        recs.append(_hit_record(tool, h, hid))
    parts.append(gem.text_part("</tool_output>"))
    return parts, recs

MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "60"))

VISIT_MAX_IMAGES = int(os.environ.get("VISIT_MAX_IMAGES", "40"))

def _img_count(msgs):
    return sum(1 for m in msgs for p in m["parts"] if p["_t"] == "img")

_AM = os.environ.get("AGENT_MODEL", "")
_IS_LOCAL = ("local" in _AM) or ("qwen3-27b" in _AM)
CTX_LIMIT = int(os.environ.get("CTX_LIMIT", "131072" if _IS_LOCAL else "900000"))
CTX_SAFE = float(os.environ.get("CTX_SAFE", "0.75"))
IMG_TOKENS = int(os.environ.get("IMG_TOKENS", "2200"))

def _est_prompt_tokens(msgs):
    chars = 0
    imgs = 0
    for m in msgs:
        for p in m["parts"]:
            if p["_t"] == "img":
                imgs += 1
            else:
                chars += len(str(p.get("text") or ""))
    return chars // 4 + imgs * IMG_TOKENS

def _fit_context(msgs):
    budget = int(CTX_LIMIT * CTX_SAFE)
    dropped = 0
    while _est_prompt_tokens(msgs) > budget and _img_count(msgs) > 2:
        _drop_oldest_images(msgs, 3)
        dropped += 3
    return dropped

def _drop_oldest_images(msgs, n_drop):
    for m in msgs:
        if n_drop <= 0:
            break
        out = []
        for p in m["parts"]:
            if p["_t"] == "img" and n_drop > 0:
                n_drop -= 1
                out.append(gem.text_part("[image dropped -- context image cap reached]"))
            else:
                out.append(p)
        m["parts"] = out

def run_react_chat(question, tools, budget=20, max_steps=26, verbose=False, extra="", rewrite=None, anchor_img=None):
    t0 = time.time()
    tdesc = "\n".join(TOOL_DESCS[t] for t in tools)
    odesc = "\n".join(OBS_DESCS[t] for t in tools)
    imgtools = [t for t in tools if t in ("search_figure", "search_page")]
    if gem.uses_native_tools():

        gem.set_tools([{"name": t, "description": TOOL_DESCS[t]} for t in tools])
        _rules = SYSTEM.split("## CRITICAL ANSWERING RULES", 1)
        _gate = SYSTEM.split("## Before you answer -- check yourself", 1)[1].split("## CRITICAL")[0]
        sys_txt = (SYSTEM_NATIVE_PROCESS.format(budget=budget)
                   + "\n## Before you answer -- check yourself" + _gate
                   + "## CRITICAL ANSWERING RULES" + _rules[1].format(tools=tdesc, obs=odesc, budget=budget))
    else:
        gem.set_tools([])
        sys_txt = SYSTEM.format(tools=tdesc, obs=odesc, budget=budget)
    if imgtools:
        sys_txt += IMG_HINT.format(imgtools=", ".join(imgtools))
    sys_txt += (extra or "")

    _uparts = []
    if anchor_img:
        try:
            _ib = open(anchor_img,"rb").read() if isinstance(anchor_img,str) else anchor_img
            _uparts.append(gem.img_bytes(_ib, "image/png"))
            _uparts.append(gem.text_part("[The figure above is an anchor figure from a paper. First read it to identify the method/entity the question refers to, then retrieve and read that paper to answer.]"))
        except Exception: pass
    _uparts.append(gem.text_part("QUESTION: " + question))
    msgs = [{"role": "system", "parts": [gem.text_part(sys_txt)]},
            {"role": "user",   "parts": _uparts}]

    searches, steps = [], []
    calls_by_tool = collections.Counter()
    n_calls, answer, bad, no_tag = 0, "", 0, 0
    failed = None
    gem.q_reset()
    prev = (0, 0, 0)

    for step in range(max_steps):
        force = n_calls >= budget
        if force:

            msgs, shed = gem.shed_images_to(msgs, keep=int(os.environ.get("FORCE_KEEP_IMAGES", "24")))
            if shed:
                print("[force] dropped %d oldest images so the final answer can be sent" % shed,
                      flush=True)
            msgs.append({"role": "user", "parts": [gem.text_part(
                "You have used all %d tool calls. Give your final answer NOW in <answer></answer> tags." % budget)]})
        try:
            txt = gem.mcall(msgs)
        except Exception as e:

            steps.append({"i": step, "raw": "", "action": "error", "error": str(e)[:300]})
            failed = str(e)[:200]
            answer = None
            break

        cur = gem.q_get()
        d_in, d_out, d_rsn = (cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2])
        prev = cur

        kind, a, b, keep = _parse(txt)

        nat = gem.last_native_parts()
        if nat and keep.startswith(txt):
            aparts = [gem.native_part(p) for p in nat]
            tail = keep[len(txt):]
            if tail:
                aparts.append(gem.text_part(tail))
        else:
            aparts = [gem.text_part(keep)]
        msgs.append({"role": "assistant", "parts": aparts})

        rec = {"i": step, "raw": txt,
               "kept": keep if keep != txt else None,
               "overrun": len(txt) - len(keep),

               "reasoning_external": _split_think(keep),
               "reasoning_internal": gem.last_reasoning(),
               "reasoning": "\n\n---- internal ----\n".join(
                   [x for x in (_split_think(keep), gem.last_reasoning()) if x.strip()]),
               "action": kind,
               "in_tok": d_in, "out_tok": d_out, "reasoning_tok": d_rsn,
               "t": round(time.time() - t0, 1)}

        if verbose:
            print("STEP %d | %s | %s" % (step, kind, (a or "")[:120]), flush=True)

        if kind == "answer" and n_calls == 0 and not force:

            rec["error"] = "answer_before_any_retrieval"
            steps.append(rec)
            msgs.append({"role": "user", "parts": [gem.text_part(
                "You have not retrieved any evidence yet -- no <tool_output> has been sent to you, so "
                "anything you just wrote came from memory, not from this corpus. Do not answer yet. "
                "Emit exactly ONE <call_tool ...>...</call_tool> and then STOP.")]})
            continue

        if kind == "answer" or force:
            answer = a if kind == "answer" else (a or answer)
            rec.update({"tool": "finish", "answer": answer, "note": "forced" if force else ""})
            steps.append(rec)
            break

        if kind == "none":

            no_tag += 1
            if no_tag >= 2:
                answer = answer or (a or "")[:2000]
                rec["error"] = "runaway_no_tag"
                steps.append(rec)
                break

        if kind == "none" or a not in tools:
            bad += 1
            rec.update({"tool": None, "note": "invalid-tag(%s)" % (a[:40] if kind != "none" else "no-tag")})
            steps.append(rec)
            if bad > 3:
                answer = a or "ERROR: no valid tag"
                break
            msgs.append({"role": "user", "parts": [gem.text_part(
                'That was not a valid action. Emit EXACTLY one of:\n'
                '  <call_tool name="%s">your query</call_tool>\n'
                '  <answer>your answer</answer>' % "|".join(tools))]})
            continue

        tool, query = a, (b or "").strip()
        if not query:
            bad += 1
            rec.update({"tool": tool, "query": "", "note": "empty-query"})
            steps.append(rec)
            msgs.append({"role": "user", "parts": [gem.text_part("Your tool call had an empty query. Try again.")]})
            continue

        query_raw = query
        query = _clean_query(query)

        rewritten = None
        if rewrite is not None and tool == "search_page":
            try:
                rq = rewrite(rec.get("reasoning") or "", query)
                if rq and rq.strip():
                    rewritten = rq.strip()
                    query = rewritten
            except Exception as e:
                print("  !! rewrite failed: %s" % str(e)[:120], flush=True)

        try:
            with _SEARCH_LOCK:
                hits = tools[tool].search(query, k=K_BY_TOOL.get(tool, 5))
            err = None
        except Exception as e:
            hits, err = [], str(e)[:200]

        if tool == "visit":
            imgs = [h for h in hits if h.get("image") is not None]
            if len(imgs) > VISIT_MAX_IMAGES:
                keep = set(id(h) for h in imgs[:VISIT_MAX_IMAGES])
                dropped = len(imgs) - VISIT_MAX_IMAGES
                hits = [h for h in hits if h.get("image") is None or id(h) in keep]
                rec["visit_pages_truncated"] = dropped

        n_calls += 1
        calls_by_tool[tool] += 1

        if err:
            rec.update({"tool": tool, "query": query, "rewritten": rewritten, "error": err, "hits": []})
            steps.append(rec)
            searches.append({"tool": tool, "query": query, "rewritten": None, "error": err, "hits": []})
            msgs.append({"role": "user", "parts": [gem.text_part("<tool_output>SEARCH ERROR: %s</tool_output>" % err)]})
            continue

        obs_parts, hit_recs = _format_obs(tool, hits, step)
        obs_parts.append(gem.text_part(
            "[tool calls used: %d/%d] If the exact value the question asks for is not visible in the "
            "results above, search again with a different query -- do NOT answer from memory."
            % (n_calls, budget)))
        rec.update({"tool": tool, "query": query, "query_raw": query_raw, "rewritten": rewritten,
                    "query_cleaned": query != query_raw, "error": None, "hits": hit_recs})
        steps.append(rec)
        searches.append({"tool": tool, "query": query, "rewritten": None, "error": None, "hits": hit_recs})
        msgs.append({"role": "user", "parts": obs_parts})
        over = _img_count(msgs) - MAX_IMAGES
        if over > 0:
            _drop_oldest_images(msgs, over)
            rec["images_dropped"] = over
        tok_dropped = _fit_context(msgs)
        if tok_dropped:
            rec["images_dropped"] = rec.get("images_dropped", 0) + tok_dropped
            print("[ctxfit] est over budget -> dropped %d oldest images" % tok_dropped, flush=True)

    qin, qout, qrsn = gem.q_get()
    return {"answer": answer, "failed": failed,
            "searches": searches,
            "steps": steps,
            "n_calls": n_calls,
            "calls_by_tool": dict(calls_by_tool),
            "in_tok": qin, "out_tok": qout, "reasoning_tok": qrsn,
            "wall_s": round(time.time() - t0, 1),
            "meta": {"harness": HARNESS_VERSION, "model": gem.AGENT_MODEL, "budget": budget,
                     "tools": list(tools), "thinking": "high", "guided_json": False,
                     "obs_retained": "all"}}
