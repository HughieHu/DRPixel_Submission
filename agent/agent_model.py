import os, json, time, random, base64, ssl, itertools, urllib.request
import re

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")
_MAXTRIES = int(os.environ.get("AGENT_MAX_RETRIES", 6))

def text_part(s):                       return {"_t": "text", "text": s}
def img_bytes(data, mime="image/jpeg"): return {"_t": "img", "data": data, "mime": mime}

BUDGET_USD = float(os.environ.get("BUDGET_USD", "1000"))
TOK = {"in": 0, "out": 0, "reasoning": 0, "calls": 0, "429": 0, "retries": 0}
import threading as _thr
_qtls = _thr.local()
def q_reset(): _qtls.d = {"in": 0, "out": 0, "reasoning": 0}
def q_get():
    d = getattr(_qtls, "d", None) or {"in": 0, "out": 0, "reasoning": 0}
    return d["in"], d["out"], d["reasoning"]
def _tok_add(i, o, rsn):
    i, o, rsn = int(i or 0), int(o or 0), int(rsn or 0)
    TOK["in"] += i; TOK["out"] += o; TOK["reasoning"] += rsn
    d = getattr(_qtls, "d", None)
    if d is None: d = {"in": 0, "out": 0, "reasoning": 0}; _qtls.d = d
    d["in"] += i; d["out"] += o; d["reasoning"] += rsn
def _is429(s):
    s = s.lower()
    return any(t in s for t in ("429", "too many requests", "rate limit", "rate_limit", "resource_exhausted", "ratelimit"))
_PRICE = {
    "gemini-3.5-flash": (0.30, 2.50), "gemini-3.1-pro-preview": (1.25, 10.0),
    "gpt-5.4": (2.50, 10.0), "gpt-5.4-mini": (0.30, 1.20), "gpt-5.4-nano": (0.10, 0.40),
    "qwen3-27b-local": (0.0, 0.0),
}
class CallFailed(Exception):
    pass

class BudgetExceeded(Exception): pass

class RateLimited(Exception):
    def __init__(self, retry_after=0, msg=""):
        super().__init__(msg or "rate limited")
        self.retry_after = retry_after or 0
def spent():
    pin, pout = _PRICE.get(AGENT_MODEL, (1.0, 5.0))
    return (TOK["in"] * pin + TOK["out"] * pout) / 1e6
def budget_guard():
    if spent() >= BUDGET_USD:
        raise BudgetExceeded("BUDGET $%.0f hit (spent $%.2f)" % (BUDGET_USD, spent()))

_QWEN_HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
_QWEN_URL = "http://%s:%s/v1/chat/completions" % (_QWEN_HOST, os.environ.get("QWEN_PORT", "8000"))

_QWEN_SCHEMA = {
    "type": "object",
    "properties": {

        "thought": {"type": "string", "maxLength": 1200},
        "action": {"type": "object",
                   "properties": {"tool": {"type": "string"},
                                  "query": {"type": "string", "maxLength": 600},
                                  "answer": {"type": "string", "maxLength": 2500}},
                   "required": ["tool"]},
    },
    "required": ["thought", "action"],
}

_GPT54_URL = os.environ.get("GPT54_URL", "")
_GPT54_KEY = os.environ.get("GPT54_KEY", "")
_AZ = {
    "gpt-5.4":         [(_GPT54_URL, _GPT54_KEY, "responses")],
    "gpt-5.4-mini":    [(_GPT54_URL, _GPT54_KEY, "responses")],
    "gpt-5.4-nano":    [(_GPT54_URL, _GPT54_KEY, "responses")],
    "qwen3-27b-local": [(_QWEN_URL, "EMPTY", "vllm")],
}
_rr = itertools.count()
try:
    import certifi; _CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _CTX = ssl.create_default_context()

def _b64(p):
    return "data:%s;base64,%s" % (p["mime"], base64.b64encode(p["data"]).decode())

def _responses_content(parts):
    out = []
    for p in parts:
        if p["_t"] == "text":
            out.append({"type": "input_text", "text": p["text"]})
        else:
            out.append({"type": "input_image", "image_url": _b64(p)})
    return out

def _chat_content(parts):
    out = []
    for p in parts:
        if p["_t"] == "text":
            out.append({"type": "text", "text": p["text"]})
        else:
            out.append({"type": "image_url", "image_url": {"url": _b64(p)}})
    return out

def _post(url, key, body, kind="responses"):
    h = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    if "api.openai.com" in url or kind == "vllm":
        h["Authorization"] = "Bearer " + key
    else:
        h["api-key"] = key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=240, context=_CTX) as resp:
        return json.load(resp)

def _azure_text(parts):
    eps = _AZ[AGENT_MODEL]
    i = next(_rr) % len(eps)
    last = ""
    for attempt in range(_MAXTRIES):
        url, key, kind = eps[(i + attempt) % len(eps)]
        budget = 8000 + 6000 * (attempt // len(eps))
        try:
            if kind == "responses":
                body = {"model": AGENT_MODEL,
                        "input": [{"role": "user", "content": _responses_content(parts)}],

                "reasoning": {"effort": "high", "summary": "auto"},
                "max_output_tokens": budget}
            elif kind == "vllm":

                body = {"model": "qwen3-27b",
                        "messages": [{"role": "user", "content": _chat_content(parts)}],
                        "max_tokens": max(budget, 24000), "temperature": 0,

                        "chat_template_kwargs": {"enable_thinking": False},
                        "response_format": {"type": "json_schema",
                                            "json_schema": {"name": "react_action", "schema": _QWEN_SCHEMA}}}
            d = _post(url, key, body, kind)
            if kind == "responses":
                txt = "".join(c.get("text", "") for o in d.get("output", [])
                              for c in o.get("content", []) if c.get("type") == "output_text")
                u = d.get("usage", {}) or {}
                _tok_add(u.get("input_tokens"), u.get("output_tokens"),
                         (u.get("output_tokens_details") or {}).get("reasoning_tokens"))
            else:
                txt = d["choices"][0]["message"].get("content") or ""
                u = d.get("usage", {}) or {}
                _tok_add(u.get("prompt_tokens"), u.get("completion_tokens"),
                         (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
            TOK["calls"] += 1
            if txt.strip():
                return txt
            last = "empty response (status=%s)" % d.get("status")
        except urllib.error.HTTPError as e:
            try: last = "%d %s" % (e.code, e.read().decode()[:300])
            except Exception: last = str(e)[:300]
            TOK["retries"] += 1
            if e.code == 429 or _is429(last): TOK["429"] += 1

            if e.code == 400 and ("reasoning" in last.lower() or "effort" in last.lower()):
                try:
                    if kind == "responses":
                        body.pop("reasoning", None)
                    else:
                        body.pop("reasoning_effort", None)
                    d = _post(url, key, body, kind)
                    txt = ("".join(c.get("text", "") for o in d.get("output", []) for c in o.get("content", []) if c.get("type") == "output_text")
                           if kind == "responses" else (d["choices"][0]["message"].get("content") or ""))
                    TOK["calls"] += 1
                    if txt.strip():
                        return txt
                except Exception as e2:
                    last = "noreason retry failed: " + str(e2)[:200]
        except Exception as e:
            last = str(e)[:300]
            TOK["retries"] += 1
            if _is429(last): TOK["429"] += 1
        time.sleep(3 + random.uniform(0, 4))
    raise RuntimeError("azure(%s) all attempts failed: %s" % (AGENT_MODEL, last))

REWRITE_MODEL  = os.environ.get("REWRITE_MODEL", "gpt-5.4")
REWRITE_EFFORT = os.environ.get("REWRITE_EFFORT", "low")
REWRITE_WORDS  = int(os.environ.get("REWRITE_WORDS", "20"))
REWRITE_STYLE  = os.environ.get("REWRITE_STYLE", "nl").lower()
_REWRITE_INSTR = (
    "You rewrite a page-retrieval search query. The corpus is whole-PAGE images of scientific papers; "
    "a dense page retriever matches on the content printed on the page. Using the agent's REASONING and "
    "its DRAFT QUERY, write ONE fluent natural-language query of about %d words that will best retrieve the "
    "single page that answers the need. Keep the specific entities, metrics, dataset/method names, table/"
    "figure numbers and values implied; write natural language, NOT a keyword bag. Output ONLY the query.")

_REWRITE_INSTR_KW = (
    "You rewrite a page-retrieval search query into a KEYWORD query (NOT a sentence). The corpus is whole-"
    "PAGE images of scientific papers. Using the agent's REASONING and its DRAFT QUERY, output about %d words "
    "as a BAG OF EXACT TERMS you expect printed ON the target page: stack precise keywords, QUOTE literal "
    "phrases / values / labels, name the Table/Figure number and the dataset / method / model names, include "
    "the metric column names. NO connective words, NO full sentences. Output ONLY the keyword query.")
_REWRITE_KW = REWRITE_STYLE.startswith("k")

def rewrite_query(thought, raw_query):
    _instr = (_REWRITE_INSTR_KW if _REWRITE_KW else _REWRITE_INSTR) % REWRITE_WORDS
    prompt = ("%s\n\nREASONING: %s\n\nDRAFT QUERY: %s\n\nYour ~%d-word %s query:"
              % (_instr, (thought or "")[:1500], (raw_query or "")[:600], REWRITE_WORDS,
                 "keyword" if _REWRITE_KW else "natural-language"))
    eps = _AZ[REWRITE_MODEL]
    i = next(_rr) % len(eps)
    for attempt in range(_MAXTRIES):
        url, key, kind = eps[(i + attempt) % len(eps)]
        try:
            body = {"model": REWRITE_MODEL,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                    "reasoning": {"effort": REWRITE_EFFORT}, "max_output_tokens": 2000}
            d = _post(url, key, body, "responses")
            txt = "".join(c.get("text", "") for o in d.get("output", [])
                          for c in o.get("content", []) if c.get("type") == "output_text")
            u = d.get("usage", {}) or {}
            _tok_add(u.get("input_tokens"), u.get("output_tokens"),
                     (u.get("output_tokens_details") or {}).get("reasoning_tokens"))
            TOK["calls"] += 1
            if txt.strip():
                return " ".join(txt.strip().split())
        except Exception as e:
            TOK["retries"] += 1
            if _is429(str(e)): TOK["429"] += 1
            time.sleep(2 + random.uniform(0, 3))
    return raw_query

import threading
_gem_tls = threading.local()
def _gemini_parts(parts):
    from google.genai import types
    out = []
    for p in parts:
        if p["_t"] == "native":
            out.append(p["part"])
        elif p["_t"] == "text":
            out.append(types.Part.from_text(text=p["text"]))
        else:
            out.append(types.Part.from_bytes(data=p["data"], mime_type=p["mime"]))
    return out

def native_part(raw):
    return {"_t": "native", "part": raw}

def last_native_parts():
    return list(getattr(_rsn_tls, "native", None) or [])

def _gem_client_get(reset=False):
    from google import genai
    from google.genai import types
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "your-gcp-project")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    if reset or getattr(_gem_tls, "client", None) is None:
        ho = types.HttpOptions(timeout=180000)
        _gem_tls.client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                       location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"), http_options=ho)
    return _gem_tls.client

def _gemini_jcall(parts, parse_retries=3):
    from google.genai import types

    real_model = AGENT_MODEL[:-6] if AGENT_MODEL.endswith("-think") else AGENT_MODEL
    if AGENT_MODEL.endswith("-think"):
        cfg = types.GenerateContentConfig(response_mime_type="application/json",
                                          thinking_config=types.ThinkingConfig(thinking_level="high"))
    else:
        cfg = types.GenerateContentConfig(response_mime_type="application/json")
    contents = _gemini_parts(parts)
    err = None
    for _ in range(parse_retries):
        delay = 5.0
        for i in range(_MAXTRIES):
            try:
                r = _gem_client_get().models.generate_content(model=real_model, contents=contents, config=cfg)
                break
            except Exception as e:
                s = str(e).lower()
                if any(t in s for t in ("429", "resource", "quota", "rate", "503", "500", "unavailable",
                                        "overload", "internal", "timeout", "deadline", "connection", "closed")) and i < _MAXTRIES - 1:
                    TOK["retries"] += 1
                    if _is429(s): TOK["429"] += 1
                    if "closed" in s: _gem_client_get(reset=True)
                    time.sleep(delay + random.uniform(0, 2)); delay = min(delay * 2, 120); continue
                raise
        try:
            um = r.usage_metadata
            _tok_add(um.prompt_token_count, um.candidates_token_count, getattr(um, "thoughts_token_count", 0))
            TOK["calls"] += 1
        except Exception:
            pass
        obj = _parse_json(r.text)
        if obj is not None:
            return obj, 0
        err = "gemini bad json"
    raise RuntimeError(err)

def _parse_json(t):
    t = (t or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json": t = t[4:]
        t = t.strip()
    i = t.find("{")
    if i < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(t[i:])[0]
    except Exception:
        try:
            return json.loads(t)
        except Exception:
            return None

def jcall(parts, parse_retries=3):
    budget_guard()
    if AGENT_MODEL.startswith("gemini"):
        return _gemini_jcall(parts, parse_retries)
    err = None
    for _ in range(parse_retries):
        txt = _azure_text(parts)
        obj = _parse_json(txt)
        if obj is not None:
            return obj, 0
        err = "no-json: " + (txt or "")[:120]
    raise RuntimeError(err or "jcall failed")

_TOOLS_SCHEMA = []

def set_tools(schema):
    global _TOOLS_SCHEMA
    _TOOLS_SCHEMA = [{"type": "function", "name": t["name"], "description": t["description"],
                      "parameters": {"type": "object",
                                     "properties": {"query": {"type": "string",
                                                              "description": "the search query"}},
                                     "required": ["query"], "additionalProperties": False}}
                     for t in schema]

def uses_native_tools():
    try:
        return _AZ[AGENT_MODEL][0][2] == "responses"
    except Exception:
        return False

def _responses_content(parts, txt_type):
    c = []
    for p in parts:
        if p["_t"] == "text":
            c.append({"type": txt_type, "text": p["text"]})
        else:
            c.append({"type": "input_image", "detail": "high", "image_url": _b64(p)})
    return c

def _responses_msgs(msgs):
    out, pending = [], None
    for m in msgs:
        parts = m["parts"]
        nat = [p["part"] for p in parts if p["_t"] == "native"]
        if m["role"] == "assistant":
            if nat:
                out.extend(nat)
                fc = [o for o in nat if o.get("type") == "function_call"]
                pending = fc[-1]["call_id"] if fc else None
            else:
                out.append({"role": "assistant",
                            "content": _responses_content(parts, "output_text")})
                pending = None
        elif m["role"] == "system":
            out.append({"role": "system", "content": _responses_content(parts, "input_text")})
        else:
            if pending:
                out.append({"type": "function_call_output", "call_id": pending,
                            "output": _responses_content(parts, "input_text")})
                pending = None
            else:
                out.append({"role": "user", "content": _responses_content(parts, "input_text")})
    return out

def _chat_msgs(msgs):
    return [{"role": m["role"], "content": _chat_content(m["parts"])} for m in msgs]

def _mcall_once(msgs, budget):
    eps = _AZ[AGENT_MODEL]
    url, key, kind = eps[0]
    if kind == "responses":
        body = {"model": AGENT_MODEL, "input": _responses_msgs(msgs),

                "reasoning": {"effort": "high", "summary": "auto"},

                "include": ["reasoning.encrypted_content"], "store": False,
                "max_output_tokens": budget}
        if _TOOLS_SCHEMA:
            body["tools"] = _TOOLS_SCHEMA
            body["parallel_tool_calls"] = False
        hdr = {"Content-Type": "application/json"}
        hdr["Authorization" if "api.openai.com" in url else "api-key"] = ("Bearer " + key) if "api.openai.com" in url else key
    elif kind == "vllm":

        body = {"model": os.environ.get("QWEN_SERVED", "qwen3-27b"),
                "messages": _chat_msgs(msgs),

                "max_tokens": int(os.environ.get("QWEN_MAX_TOKENS", "8000")),
                "temperature": float(os.environ.get("QWEN_TEMP", "0.6")),
                "top_p": float(os.environ.get("QWEN_TOP_P", "0.95")),
                "top_k": int(os.environ.get("QWEN_TOP_K", "20")),
                "presence_penalty": float(os.environ.get("QWEN_PRES_PEN", "1.0")),
                "seed": int(os.environ.get("QWEN_SEED", "0")),

                "chat_template_kwargs": {"enable_thinking":
                                         os.environ.get("QWEN_THINK", "1") == "1"},
                "stop": ["</call_tool>", "</answer>"], "include_stop_str_in_output": True}
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    else:
        body = {"model": AGENT_MODEL, "messages": _chat_msgs(msgs), "max_tokens": budget}
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + key}

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=1200, context=_CTX) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:

        try:
            detail = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            detail = ""
        raise RuntimeError("HTTP %s from %s: %s" % (e.code, kind, detail)) from None

    if kind == "responses":
        u = d.get("usage") or {}
        _tok_add(u.get("input_tokens"), u.get("output_tokens"),
                 (u.get("output_tokens_details") or {}).get("reasoning_tokens"))
        TOK["calls"] += 1
        txt, rsn = "", ""
        for item in (d.get("output") or []):
            if item.get("type") == "reasoning":
                for sm in (item.get("summary") or []):
                    rsn += (sm.get("text") or "")
            for c in (item.get("content") or []):
                if c.get("type") in ("output_text", "text"):
                    txt += c.get("text") or ""

        items = d.get("output") or []
        fc = [o for o in items if o.get("type") == "function_call"]
        _set_reasoning(rsn)
        _rsn_tls.native = items
        if fc:
            try:
                q = json.loads(fc[0].get("arguments") or "{}").get("query", "")
            except Exception:
                q = ""

            mt = re.search(r"<think>.*?</think>", txt or "", re.S | re.I)
            return '%s<call_tool name="%s">%s</call_tool>' % (
                mt.group(0) if mt else "", fc[0].get("name", ""), q)
        if txt and "<answer>" not in txt:
            body_, think_ = txt, ""
            if "</think>" in txt:
                think_, body_ = txt.split("</think>", 1)
                think_ += "</think>"
            txt = think_ + "<answer>%s</answer>" % body_.strip()
        return txt
    u = d.get("usage") or {}
    _tok_add(u.get("prompt_tokens"), u.get("completion_tokens"),
             (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    TOK["calls"] += 1
    m = (d.get("choices") or [{}])[0].get("message") or {}

    _set_reasoning(m.get("reasoning_content") or "")
    return (m.get("content") or "") or (m.get("reasoning_content") or "")

def _gemini_mcall(msgs, budget):
    from google.genai import types
    real = AGENT_MODEL[:-6] if AGENT_MODEL.endswith("-think") else AGENT_MODEL
    sysm = "\n".join(p["text"] for m in msgs if m["role"] == "system"
                     for p in m["parts"] if p["_t"] == "text")
    contents = [types.Content(role=("model" if m["role"] == "assistant" else "user"),
                              parts=_gemini_parts(m["parts"]))
                for m in msgs if m["role"] != "system"]
    cfg = types.GenerateContentConfig(
        system_instruction=sysm or None,
        thinking_config=types.ThinkingConfig(thinking_level="high", include_thoughts=True),
        stop_sequences=["</call_tool>"],
        max_output_tokens=budget)
    delay = 5.0
    for i in range(_MAXTRIES):
        try:
            r = _gem_client_get().models.generate_content(model=real, contents=contents, config=cfg)
            break
        except Exception as e:
            s = str(e).lower()
            if i < _MAXTRIES - 1 and any(t in s for t in ("429", "resource", "quota", "rate", "503",
                                                          "500", "unavailable", "deadline", "closed")):
                TOK["retries"] += 1
                if _is429(s): TOK["429"] += 1
                if "closed" in s: _gem_client_get(reset=True)
                time.sleep(delay + random.uniform(0, 2)); delay = min(delay * 2, 120); continue
            raise
    try:
        um = r.usage_metadata
        _tok_add(um.prompt_token_count, um.candidates_token_count,
                 getattr(um, "thoughts_token_count", 0))
        TOK["calls"] += 1
    except Exception:
        pass
    rsn, vis, native = "", "", []
    try:
        for c in (r.candidates or []):
            for p in (c.content.parts or []):
                native.append(p)
                t = getattr(p, "text", "") or ""
                if getattr(p, "thought", False):
                    rsn += t
                else:
                    vis += t
    except Exception:
        pass
    _set_reasoning(rsn)
    _rsn_tls.native = native

    return vis or (r.text or "")

_CTX_FULL = ("maximum context length", "exceeds model", "context_length_exceeded",
             "reduce the length", "too many tokens", "longer than the maximum")

def _is_ctx_full(msg):
    m = (msg or "").lower()
    return any(k in m for k in _CTX_FULL)

def _msg_parts(m):
    return m.get("parts") if isinstance(m.get("parts"), list) else None

def _img_index(msgs):
    return [(i, j) for i, m in enumerate(msgs)
            for j, p in enumerate(_msg_parts(m) or [])
            if isinstance(p, dict) and p.get("_t") == "img"]

def _rebuild(msgs, drop):
    out = []
    for i, m in enumerate(msgs):
        ps = _msg_parts(m)
        if ps is None:
            out.append(m)
            continue
        mm = dict(m)
        mm["parts"] = [text_part("[earlier page image dropped: context limit]")
                       if (i, j) in drop else p
                       for j, p in enumerate(ps)]
        out.append(mm)
    return out

def _drop_oldest_images(msgs, drop_frac=0.35):
    idx = _img_index(msgs)
    if not idx:
        return msgs, 0
    ndrop = max(1, int(len(idx) * drop_frac))
    return _rebuild(msgs, set(idx[:ndrop])), ndrop

def shed_images_to(msgs, keep=24):
    idx = _img_index(msgs)
    if len(idx) <= keep:
        return msgs, 0
    ndrop = len(idx) - keep
    return _rebuild(msgs, set(idx[:ndrop])), ndrop

def mcall(msgs, retries=3, transient_retries=12):
    budget_guard()
    last = ""
    tr = 0
    a = 0
    ctx_shed = 0
    while a < retries:
        budget = 8000 + 8000 * a
        try:
            txt = _gemini_mcall(msgs, budget) if AGENT_MODEL.startswith("gemini") \
                  else _mcall_once(msgs, budget)
        except Exception as e:
            msg = str(e)

            if _is_ctx_full(msg) and ctx_shed < 3:
                msgs, nd = _drop_oldest_images(msgs)
                if nd:
                    ctx_shed += 1
                    TOK["ctx_shed"] = TOK.get("ctx_shed", 0) + nd
                    print("[ctx] overflow -> dropped %d oldest images (shed %d/3)" % (nd, ctx_shed),
                          flush=True)
                    continue
            is_transient = (
                isinstance(e, RateLimited)
                or "429" in msg
                or "timed out" in msg.lower()
                or "urlopen error" in msg
                or "Remote end closed" in msg
                or "Connection reset" in msg
                or "HTTP 500" in msg or "HTTP Error 500" in msg
                or "HTTP 503" in msg or "HTTP Error 503" in msg
            )
            if is_transient:
                tr += 1
                TOK["retries"] += 1
                if "429" in msg or isinstance(e, RateLimited):
                    TOK["429"] += 1
                if tr > transient_retries:
                    raise CallFailed("transient failure x%d: %s" % (tr, msg[:140]))
                ra = getattr(e, "retry_after", 0)
                time.sleep((ra or min(10 * tr, 60)) + random.uniform(0, 3))
                continue
            last = "call failed: " + msg[:160]
            TOK["retries"] += 1
            a += 1
            time.sleep(3 + 3 * a)
            continue
        if (txt or "").strip():
            return txt
        last = "empty reply"
        TOK["retries"] += 1
        a += 1
    raise CallFailed(last or "mcall failed")

_rsn_tls = threading.local()
def _set_reasoning(s):
    _rsn_tls.s = s or ""
    _rsn_tls.native = []
def last_reasoning():   return getattr(_rsn_tls, "s", "") or ""

