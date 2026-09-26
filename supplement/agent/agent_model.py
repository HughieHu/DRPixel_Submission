"""Multi-backend agent model for the ReAct retrieval experiment. Same neutral interface as the
old gem.py (text_part / img_bytes / jcall / spent), so react.py just imports this instead.
Backend chosen by env AGENT_MODEL:
  gemini-3.5-flash         -> Google Vertex AI (google-genai)
  gpt-5.4 / -mini / -nano  -> Azure OpenAI *responses* API   (reasoning effort = high)
  grok-4.3                 -> Azure AI services *chat/completions* API (reasoning_effort = high)
Each Azure model has TWO (endpoint,key) pairs for 429 avoidance -> round-robin + failover.
Parts are neutral dicts ({_t:text|img}) so the SAME ReAct loop drives every model (incl. images)."""
import os, json, time, random, base64, ssl, itertools, urllib.request
import re

AGENT_MODEL = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")
_MAXTRIES = int(os.environ.get("AGENT_MAX_RETRIES", 6))

# ---- neutral parts (backend-agnostic) ----
def text_part(s):                       return {"_t": "text", "text": s}
def img_bytes(data, mime="image/jpeg"): return {"_t": "img", "data": data, "mime": mime}

# ---- token accounting + budget guard ----
BUDGET_USD = float(os.environ.get("BUDGET_USD", "1000"))
TOK = {"in": 0, "out": 0, "reasoning": 0, "calls": 0, "429": 0, "retries": 0}
import threading as _thr
_qtls = _thr.local()   # per-question (per-thread) token accumulator, concurrency-safe
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
_PRICE = {  # rough per-1M (in, out) just for the $ guard; not billing-exact
    "gemini-3.5-flash": (0.30, 2.50), "gemini-3.1-pro-preview": (1.25, 10.0),
    "gpt-5.4": (2.50, 10.0), "gpt-5.4-mini": (0.30, 1.20), "gpt-5.4-nano": (0.10, 0.40),
    "grok-4.3": (3.0, 15.0), "deepseek-ai/DeepSeek-V4-Pro": (1.0, 3.0),
    "qwen3-27b-local": (0.0, 0.0),   # self-hosted vLLM on Torch -> no $ cost
    "qwen3-27b-sft": (0.0, 0.0),
    "qwen3-35b-a3b-local": (0.0, 0.0),   # self-hosted vLLM (Qwen3.6-35B-A3B MoE) -> no $ cost
}
class CallFailed(Exception):
    """The provider never answered. The caller must NOT turn this into an `answer` string."""


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

# ============================ Azure backends (gpt responses / grok chat) ============================
_K1 = os.environ.get("AZURE_KEY_A", "")
_K2 = os.environ.get("AZURE_KEY_B", "")
_KG = os.environ.get("AZURE_KEY_GROK", "")
_RESP_A = os.environ.get("AZURE_RESPONSES_URL_A", "")
_RESP_B = os.environ.get("AZURE_RESPONSES_URL_B", "")
_GROK_A = os.environ.get("AZURE_GROK_URL_A", "")
_GROK_B = os.environ.get("AZURE_GROK_URL_B", "")
# model -> [(url, key, kind)]
_TOGETHER = "https://api.together.xyz/v1/chat/completions"
_TKEY = os.environ.get("TOGETHER_API_KEY", "")
# self-hosted Qwen3.6-27B (VL) via vLLM OpenAI server on Torch; QWEN_HOST = server node hostname
_QWEN_HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
_QWEN_URL = "http://%s:%s/v1/chat/completions" % (_QWEN_HOST, os.environ.get("QWEN_PORT", "8000"))
# Qwen thinking-mode emits prose, never the ReAct JSON -> force valid JSON via vLLM guided decoding.
# Schema admits BOTH action shapes: search {tool,query} and finish {tool,answer}. tool required; the
# rest optional so either form validates. enable_thinking=false avoids the auto-<think> vs guided clash.
_QWEN_SCHEMA = {
    "type": "object",
    "properties": {
        # maxLength caps the thought so verbose models (e.g. 35B-A3B) can't write multi-thousand-token
        # thoughts that overflow max_tokens (truncated JSON) and balloon the accumulated trace past context.
        "thought": {"type": "string", "maxLength": 1200},
        "action": {"type": "object",
                   "properties": {"tool": {"type": "string"},
                                  "query": {"type": "string", "maxLength": 600},
                                  "answer": {"type": "string", "maxLength": 2500}},
                   "required": ["tool"]},
    },
    "required": ["thought", "action"],
}
# GPT54_URL/GPT54_KEY point gpt-5.4 at the official OpenAI API (kelly Azure is dead).
_GPT54_URL = os.environ.get("GPT54_URL", _RESP_B)
_GPT54_KEY = os.environ.get("GPT54_KEY", _K2)
_AZ = {
    "gpt-5.4":      [(_GPT54_URL, _GPT54_KEY, "responses")],
    "gpt-5.4-mini": [(_RESP_B, _K2, "responses")],
    "gpt-5.4-nano": [(_RESP_B, _K2, "responses")],
    "grok-4.3":     [(_GROK_A, _K1, "chat"), (_GROK_B, _KG, "chat")],
    "deepseek-ai/DeepSeek-V4-Pro": [(_TOGETHER, _TKEY, "together")],   # Together API, single key, Bearer auth
    "qwen3-27b-local": [(_QWEN_URL, "EMPTY", "vllm")],                 # local vLLM, OpenAI chat, no real auth
    "qwen3-27b-sft": [(_QWEN_URL, "EMPTY", "vllm")],                   # merged SFT model on same endpoint
    "qwen3-35b-a3b-local": [(_QWEN_URL, "EMPTY", "vllm")],             # local vLLM Qwen3.6-35B-A3B MoE, same served-name label
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
    h = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}   # Together's Cloudflare blocks default UA
    if "api.openai.com" in url or kind in ("together", "vllm", "openrouter", "fireworks"):
        h["Authorization"] = "Bearer " + key
    else:
        h["api-key"] = key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    with urllib.request.urlopen(req, timeout=240, context=_CTX) as resp:
        return json.load(resp)

def _azure_text(parts):
    """Return assistant text. Round-robin the 2 endpoints, failover on error, grow token budget
    if the (reasoning) response came back truncated/empty. reasoning effort = high."""
    eps = _AZ[AGENT_MODEL]
    i = next(_rr) % len(eps)
    last = ""
    for attempt in range(_MAXTRIES):
        url, key, kind = eps[(i + attempt) % len(eps)]
        budget = 8000 + 6000 * (attempt // len(eps))      # bump budget on later rounds
        try:
            if kind == "responses":
                body = {"model": AGENT_MODEL,
                        "input": [{"role": "user", "content": _responses_content(parts)}],
                        # summary="auto" is REQUIRED to get the reasoning TEXT back; with effort alone the
                # summary array comes back empty and steps[i].reasoning is blank for every GPT run.
                "reasoning": {"effort": "high", "summary": "auto"},
                "max_output_tokens": budget}
            elif kind == "together":   # DeepSeek via Together: OpenAI chat, Bearer, no reasoning param
                body = {"model": AGENT_MODEL,
                        "messages": [{"role": "user", "content": _chat_content(parts)}], "max_tokens": budget}
            elif kind == "vllm":       # local vLLM (Qwen3.6 VL): OpenAI chat, served-model-name, temp 0
                # 35B-A3B writes very long "thought" values -> 8000 truncates the JSON. Give it room
                # to close the JSON (guided decoding needs to reach the closing brace).
                body = {"model": "qwen3-27b",
                        "messages": [{"role": "user", "content": _chat_content(parts)}],
                        "max_tokens": max(budget, 24000), "temperature": 0,
                        # force ReAct JSON (thinking-mode otherwise returns prose -> no-json); no <think>
                        "chat_template_kwargs": {"enable_thinking": False},
                        "response_format": {"type": "json_schema",
                                            "json_schema": {"name": "react_action", "schema": _QWEN_SCHEMA}}}
            else:                      # grok chat
                body = {"model": AGENT_MODEL,
                        "messages": [{"role": "user", "content": _chat_content(parts)}],
                        "reasoning_effort": "high", "max_tokens": budget}
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
            # drop reasoning param if the model rejects it, retry same endpoint
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

# ============================ query rewriter (a SEPARATE gpt-5.4) ============================
# For the page_query_rewrite experiment: before each real retrieve, a dedicated gpt-5.4 turns the
# agent's (thought, draft query) into ONE ~20-word natural-language query. Independent of AGENT_MODEL
# (always gpt-5.4). Falls back to the raw query on failure so a run never breaks. Tokens bill normally.
REWRITE_MODEL  = os.environ.get("REWRITE_MODEL", "gpt-5.4")
REWRITE_EFFORT = os.environ.get("REWRITE_EFFORT", "low")   # a rewrite needs little reasoning
REWRITE_WORDS  = int(os.environ.get("REWRITE_WORDS", "20"))
REWRITE_STYLE  = os.environ.get("REWRITE_STYLE", "nl").lower()   # "nl" (natural language) | "keyword"
_REWRITE_INSTR = (
    "You rewrite a page-retrieval search query. The corpus is whole-PAGE images of scientific papers; "
    "a dense page retriever matches on the content printed on the page. Using the agent's REASONING and "
    "its DRAFT QUERY, write ONE fluent natural-language query of about %d words that will best retrieve the "
    "single page that answers the need. Keep the specific entities, metrics, dataset/method names, table/"
    "figure numbers and values implied; write natural language, NOT a keyword bag. Output ONLY the query.")
# keyword variant: SAME length target, but a bag of exact terms instead of a fluent sentence.
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
                return " ".join(txt.strip().split())          # collapse to one clean line
        except Exception as e:
            TOK["retries"] += 1
            if _is429(str(e)): TOK["429"] += 1
            time.sleep(2 + random.uniform(0, 3))
    return raw_query                                          # fail-safe: never break the run

# ============================ Gemini backend (Vertex) ============================
import threading
_gem_tls = threading.local()      # per-thread Vertex client (shared client races under concurrency)
def _gemini_parts(parts):
    from google.genai import types
    out = []
    for p in parts:
        if p["_t"] == "native":            # a raw types.Part echoed back verbatim (carries thought_signature)
            out.append(p["part"])
        elif p["_t"] == "text":
            out.append(types.Part.from_text(text=p["text"]))
        else:
            out.append(types.Part.from_bytes(data=p["data"], mime_type=p["mime"]))
    return out


def native_part(raw):
    """Wrap a provider-native part so it survives a round-trip through our msgs format."""
    return {"_t": "native", "part": raw}


def last_native_parts():
    """The model turn EXACTLY as the provider returned it (thought parts + thought_signature).

    Gemini 3 encrypts its reasoning state into part.thought_signature. In stateless mode the docs
    require echoing those parts back verbatim, or the model's reasoning restarts from scratch every
    turn -- which is what made it re-issue near-identical queries 20 times in a row.
    """
    return list(getattr(_rsn_tls, "native", None) or [])

def _gem_client_get(reset=False):
    from google import genai
    from google.genai import types
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/work/literatureIR/genbench/gemini-vertex-key.json")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "your-gcp-project")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    if reset or getattr(_gem_tls, "client", None) is None:
        ho = types.HttpOptions(timeout=180000)
        _gem_tls.client = genai.Client(vertexai=True, project=os.environ["GOOGLE_CLOUD_PROJECT"],
                                       location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"), http_options=ho)
    return _gem_tls.client

def _gemini_jcall(parts, parse_retries=3):
    from google.genai import types
    # alias "<model>-think" -> same gemini model but forced high thinking (default is dynamic/unset).
    # the alias only changes the result filename; the API must get the REAL model id.
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
                    if "closed" in s: _gem_client_get(reset=True)   # rebuild this thread's client
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

# ============================ shared JSON parse + dispatch ============================
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
    """Unified: parts -> parsed JSON dict. Returns (dict, 0)."""
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


# ==================== MULTI-TURN CHAT (DR-Tulu harness: react_chat.py) ====================
# msgs = [{"role": "system"|"user"|"assistant", "parts": [text_part/img_bytes, ...]}]
# Free-form output (tags parsed by regex) -> NO guided JSON -> native thinking stays ON.

# ---- PLAN B: the GPT-5.4 family cannot take a stop sequence (reasoning models reject `stop`), so
# left to free-text tags it writes out the WHOLE ReAct loop in one generation -- inventing its own
# <tool_output> blocks with fake doc ids and then "answering from them" (n_calls=0, pure parametric
# recall). Native function calling makes the API itself HALT at the tool call, exactly like gemini's
# stop_sequences and vLLM's `stop`. Verified on our endpoint: generation stops at function_call,
# function_call_output accepts a real page IMAGE, and the model reads it (1464 vs 79 ALMs, correct).
_TOOLS_SCHEMA = []


def set_tools(schema):
    """schema: [{"name":..., "description":...}] -- registered once per run by the harness."""
    global _TOOLS_SCHEMA
    _TOOLS_SCHEMA = [{"type": "function", "name": t["name"], "description": t["description"],
                      "parameters": {"type": "object",
                                     "properties": {"query": {"type": "string",
                                                              "description": "the search query"}},
                                     "required": ["query"], "additionalProperties": False}}
                     for t in schema]


def uses_native_tools():
    """True for providers where the tool call is a real API item, not a text tag."""
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
    """Our provider-neutral msgs -> Responses API input items.

    An assistant turn that carries native items (reasoning + function_call) is spliced back in
    VERBATIM -- that is what preserves GPT's reasoning across turns. The user turn that follows a
    function_call becomes a function_call_output whose `output` is a content array, so the retrieved
    PAGE IMAGES ride back to the model through the tool channel.
    """
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
            if pending:                       # this is a tool result -> send it through the tool channel
                out.append({"type": "function_call_output", "call_id": pending,
                            "output": _responses_content(parts, "input_text")})
                pending = None
            else:
                out.append({"role": "user", "content": _responses_content(parts, "input_text")})
    return out


def _chat_msgs(msgs):
    return [{"role": m["role"], "content": _chat_content(m["parts"])} for m in msgs]


def _mcall_once(msgs, budget):
    """one provider call -> assistant text. Raises on failure."""
    eps = _AZ[AGENT_MODEL]
    url, key, kind = eps[0]
    if kind == "responses":                                   # Azure GPT-5.4 family
        body = {"model": AGENT_MODEL, "input": _responses_msgs(msgs),
                # summary="auto" is what actually returns the reasoning TEXT. effort alone gives a
                # reasoning_tokens COUNT and an empty summary[] -- steps[i].reasoning stays blank.
                "reasoning": {"effort": "high", "summary": "auto"},
                # The echoed `reasoning` item comes back with content=[] and NO encrypted payload unless
                # we ask for it -- so without this, GPT's hidden CoT is DROPPED between turns and its
                # reasoning restarts from scratch every step. Gemini gets its reasoning carried across
                # turns via thought_signature; without `include` here, GPT would not, and the two models
                # would be running under different conditions in the main table.
                # store=False keeps us stateless (no server-side retention): the encrypted blob rides
                # in the request, exactly like gemini's signature.
                "include": ["reasoning.encrypted_content"], "store": False,
                "max_output_tokens": budget}
        if _TOOLS_SCHEMA:
            body["tools"] = _TOOLS_SCHEMA
            body["parallel_tool_calls"] = False  # __NO_PARALLEL_TOOLCALLS__
        hdr = {"Content-Type": "application/json"}
        hdr["Authorization" if "api.openai.com" in url else "api-key"] = ("Bearer " + key) if "api.openai.com" in url else key
    elif kind == "vllm":                                      # local Qwen via vLLM (OpenAI chat)
        # Qwen3 thinking mode: the model card says "DO NOT use greedy decoding" -- it "can lead to
        # performance degradation and endless repetitions". With QWEN_TEMP=0.0 we hit exactly that:
        # a 24k-token unterminated <think> block, repeated verbatim on every retry (103k output
        # tokens and 43 minutes on ONE question). These are the sampling params Qwen ships for
        # thinking mode, plus a presence penalty to curb repetition.
        body = {"model": os.environ.get("QWEN_SERVED", "qwen3-27b"),
                "messages": _chat_msgs(msgs),
                # A legitimate qwen step spends 88-1819 output tokens (longest real answer seen: 1819).
                # max_tokens was pinned at 24000, so every runaway <think> ran to the full 24k before
                # being cut: ONE question burned 49,829 tokens and 24 minutes on two of them. 8000
                # leaves 4x headroom over the longest real answer and caps a runaway at a third of the
                # cost. Only the CHAT harness -- react.py's legacy path above is untouched.
                "max_tokens": int(os.environ.get("QWEN_MAX_TOKENS", "8000")),
                "temperature": float(os.environ.get("QWEN_TEMP", "0.6")),
                "top_p": float(os.environ.get("QWEN_TOP_P", "0.95")),
                "top_k": int(os.environ.get("QWEN_TOP_K", "20")),
                "presence_penalty": float(os.environ.get("QWEN_PRES_PEN", "1.0")),
                "seed": int(os.environ.get("QWEN_SEED", "0")),      # sampling, but reproducible
                # QWEN_THINK=0 turns off Qwen's NATIVE thinking mode. The prompt still asks for a
                # visible <think> block, so we keep the explicit, trainable reasoning channel while
                # dropping the native mode's verbosity (base: 1128 output tokens/step with it on,
                # vs ~200/step in the old guided-JSON harness -- and decode is 98% of the wall time).
                "chat_template_kwargs": {"enable_thinking":
                                         os.environ.get("QWEN_THINK", "1") == "1"},
                "stop": ["</call_tool>", "</answer>"], "include_stop_str_in_output": True}
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}
    else:                                                     # together / plain chat
        body = {"model": AGENT_MODEL, "messages": _chat_msgs(msgs), "max_tokens": budget}
        hdr = {"Content-Type": "application/json", "Authorization": "Bearer " + key}

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=hdr)
    try:
        with urllib.request.urlopen(req, timeout=1200, context=_CTX) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # str(HTTPError) is only "HTTP Error 400: Bad Request" -- the REASON is in the body. Without
        # it, a deterministic context overflow ("Input length (131516) exceeds model's maximum
        # context length (131072)") was indistinguishable from a transient blip, and 39 questions
        # retried forever against a request that could never succeed.
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
        # Re-render the native items as the DR-Tulu tag form, so react_chat's parser and the LOGGED
        # trajectory are byte-identical in shape across all three providers.
        items = d.get("output") or []
        fc = [o for o in items if o.get("type") == "function_call"]
        _set_reasoning(rsn)
        _rsn_tls.native = items                       # echoed back verbatim next turn
        if fc:
            try:
                q = json.loads(fc[0].get("arguments") or "{}").get("query", "")
            except Exception:
                q = ""
            # Keep ONLY the <think> the model wrote. Alongside the real function_call it also emits a
            # stray '<tool_call> {"query": ...}' blob in the message; letting that into the history
            # means it conditions on -- and imitates -- its own malformed tag on the next turn.
            # NEVER fall back to the internal summary here: that would make the hidden channel
            # masquerade as the written one and destroy the external/internal distinction.
            mt = re.search(r"<think>.*?</think>", txt or "", re.S | re.I)
            return '%s<call_tool name="%s">%s</call_tool>' % (
                mt.group(0) if mt else "", fc[0].get("name", ""), q)
        if txt and "<answer>" not in txt:
            body_, think_ = txt, ""
            if "</think>" in txt:                      # the model wrote its own <think> before answering
                think_, body_ = txt.split("</think>", 1)
                think_ += "</think>"
            txt = think_ + "<answer>%s</answer>" % body_.strip()
        return txt
    u = d.get("usage") or {}
    _tok_add(u.get("prompt_tokens"), u.get("completion_tokens"),
             (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
    TOK["calls"] += 1
    m = (d.get("choices") or [{}])[0].get("message") or {}
    # vLLM with a reasoning parser splits the <think> block out into reasoning_content
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
        stop_sequences=["</call_tool>"],   # DR-Tulu: generation halts at the tool call; we inject the REAL output
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
                    rsn += t                      # thought summary (needs include_thoughts=True)
                else:
                    vis += t                      # the text the harness parses
    except Exception:
        pass
    _set_reasoning(rsn)
    _rsn_tls.native = native
    # VERBATIM. Any <think> in here is one the MODEL chose to write -- we never fabricate one from the
    # internal summary, because that would make the hidden channel masquerade as the written one and
    # destroy the exact distinction we are trying to measure (does the model write its reasoning down,
    # where it persists and can be re-read, or does it keep it hidden?).
    # What we SEND back is still the native parts, which carry thought_signature.
    return vis or (r.text or "")


_CTX_FULL = ("maximum context length", "exceeds model", "context_length_exceeded",
             "reduce the length", "too many tokens", "longer than the maximum")


def _is_ctx_full(msg):
    m = (msg or "").lower()
    return any(k in m for k in _CTX_FULL)


def _msg_parts(m):
    return m.get("parts") if isinstance(m.get("parts"), list) else None


def _img_index(msgs):
    """[(msg_i, part_j)] of every image part, oldest first."""
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
    """Shed the OLDEST image parts (leaving a text placeholder); all real text is kept.

    A 20-tool-call trajectory that retains every observation overflows qwen3-27b's 131072-token
    window. Losing the whole question to that is far worse than losing the earliest page images --
    the ones the model has already had many turns to read. Returns (msgs, n_dropped); the caller
    logs n_dropped, so a drop is never silent.
    """
    idx = _img_index(msgs)
    if not idx:
        return msgs, 0
    ndrop = max(1, int(len(idx) * drop_frac))
    return _rebuild(msgs, set(idx[:ndrop])), ndrop


def shed_images_to(msgs, keep=24):
    """Keep only the `keep` most recent images. Used on the forced-answer turn, where the request
    would otherwise carry every image ever retrieved and could not be sent at all."""
    idx = _img_index(msgs)
    if len(idx) <= keep:
        return msgs, 0
    ndrop = len(idx) - keep
    return _rebuild(msgs, set(idx[:ndrop])), ndrop

def mcall(msgs, retries=3, transient_retries=12):
    """Multi-turn call -> assistant text.

    A saturated vLLM (or an Azure TPM wall) is a TRANSIENT failure: back off and retry, do not give
    up after 3 tries and do not ever return the error text as the model's answer.
    """
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
            # A context overflow is DETERMINISTIC: retrying the same request can never work.
            # Shed the oldest images and try again -- an answer from recent evidence beats no answer.
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
                continue                      # a transient failure does not consume a real retry
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




# ---- capture the provider's hidden reasoning (per thread, for full-trajectory logging) ----
_rsn_tls = threading.local()
def _set_reasoning(s):
    _rsn_tls.s = s or ""
    _rsn_tls.native = []           # cleared by default; the gemini path re-populates it
def last_reasoning():   return getattr(_rsn_tls, "s", "") or ""


# __OPENAI_OFFICIAL_PATCH__
import os as _os
if _os.environ.get("GPT54_URL") and _os.environ.get("GPT54_KEY"):
    _u = _os.environ["GPT54_URL"]; _k = _os.environ["GPT54_KEY"]
    _AZ["gpt-5.4"] = [(_u, _k, "responses")]
    _AZ["gpt-5.4-mini"] = [(_u, _k, "responses")]
    _AZ["gpt-5.4-nano"] = [(_u, _k, "responses")]
    print("[agent_model] gpt-5.4 AGENT -> official OpenAI:", _u.split("//")[-1][:28])
