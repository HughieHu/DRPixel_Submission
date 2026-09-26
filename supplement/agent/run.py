"""Run ONE config over the bench (100 q). Builds the config's retrievers (index+encoder on GPU),
runs the gemini ReAct agent per question (<=budget search calls), grades answer-vs-gold + computes
retrieval-hit, checkpoints per question (resumable), reports accuracy by modality.
  python run.py --config txtfig_q8 [--limit N] [--budget 20]
Run the 7 configs on 7 GPUs (one each).
"""
import argparse, json, os, time
import configs, retriever, react, judge
import os as _os
HARNESS = _os.environ.get("HARNESS", "react")   # 'react' (single-turn rebuild) | 'chat' (DR-Tulu multi-turn)
if HARNESS == "chat":
    import react_chat as _rc

import agent_model as gem

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=list(configs.CONFIGS))
_VLDR = os.environ.get("VLDR_ROOT", "/path/to/vldr")   # Berkeley: /path/to/vldr
ap.add_argument("--bench", default=_VLDR + "/experiment/bench_100_clean.json")
ap.add_argument("--out", default=None)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--budget", type=int, default=20)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

cfg = configs.CONFIGS[args.config]
bench = json.load(open(args.bench))
if args.limit:
    bench = bench[:args.limit]
_model = os.environ.get("AGENT_MODEL", "gemini-3.5-flash")
_model_safe = _model.replace("/", "_")   # model names like deepseek-ai/DeepSeek-V4-Pro -> safe filename
out = args.out or _VLDR + "/experiment/results/%s__%s.json" % (_model_safe, args.config)
os.makedirs(os.path.dirname(out), exist_ok=True)
print("AGENT_MODEL=%s | out=%s" % (_model, out), flush=True)

# resume: skip qids already in the out file
done = {}
if os.path.exists(out):
    try:
        prev = json.load(open(out))
        for r in (prev.get("results", prev) if isinstance(prev, dict) else prev):
            # A question whose provider call FAILED is not done -- this run retries it.
            if r.get("failed") or not (r.get("answer") or "").strip() or \
               (r.get("answer") or "").startswith("ERROR"):
                continue
            done[r["_qid"]] = r
    except Exception:
        pass
print("config=%s tools=%s | %d questions (%d already done)" % (args.config, cfg, len(bench), len(done)), flush=True)

print("building retrievers...", flush=True)
tools = {tn: retriever.make(par, args.device) for tn, par in cfg.items()}

# concurrent question processing -- each question runs the ReAct agent independently; the slow part
# is the model API (parallelized), GPU retrieval is serialized inside react via _SEARCH_LOCK.
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
CONC = int(os.environ.get("CONCURRENCY", "4"))
results = list(done.values())                 # already-done carried over
todo = [it for it in bench if it["_qid"] not in done]
t0 = time.time(); _lock = threading.Lock(); _n = [0]
print("processing %d questions, CONCURRENCY=%d" % (len(todo), CONC), flush=True)

_extra = react.PROMPT_EXTRA.get(args.config, "")   # per-config prompt augmentation (one-shot / good-query)
_rewrite = gem.rewrite_query if args.config in getattr(configs, "REWRITE_CONFIGS", set()) else None
if _rewrite:
    print("QUERY REWRITE ON: separate %s rewrites each search_page query -> ~%sw NL query"
          % (gem.REWRITE_MODEL, gem.REWRITE_WORDS), flush=True)
def _process(item):
    _fn = _rc.run_react_chat if HARNESS == "chat" else react.run_react
    _kw = {"anchor_img": item.get("anchor_img")} if HARNESS == "chat" else {}
    res = _fn(item["question"], tools, budget=args.budget, max_steps=args.budget + 6, extra=_extra, rewrite=_rewrite, **_kw)  # max_steps tracks budget (>25 fix) + anchor_img (M3)
    return {"_qid": item["_qid"], "modality": item["modality"], "answer_type": item["answer_type"],
            "doc": item["doc"],
            "question": item.get("question"), "gold_answer": item.get("gold_answer"),   # self-contained record
            "answer": res["answer"], "failed": res.get("failed"), "correct": None,
            "reason": "pending gpt-5.4-mini eval",
            # ---- cost (Table 2/3 of the deck) ----
            "n_calls": res["n_calls"], "calls_by_tool": res.get("calls_by_tool"),
            "in_tok": res.get("in_tok"), "out_tok": res.get("out_tok"), "reasoning_tok": res.get("reasoning_tok"),
            "wall_s": res.get("wall_s"),
            # ---- retrieval + FULL trajectory (rebuild every step: what the model said, what came back) ----
            "hits": judge.retrieval_hits(item, res["searches"]), "searches": res["searches"],
            "steps": res.get("steps"), "meta": res.get("meta")}

with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = [ex.submit(_process, it) for it in todo]
    for fut in as_completed(futs):
        rec = fut.result()
        with _lock:
            results.append(rec); _n[0] += 1
            if _n[0] % 5 == 0 or _n[0] == len(todo):
                json.dump({"config": args.config, "results": results}, open(out, "w"), indent=1)  # checkpoint
                print("  %d/%d | $%.2f | 429=%d | %.0fs" % (_n[0], len(todo), gem.spent(),
                                                            gem.TOK.get("429", 0), time.time() - t0), flush=True)

# aggregate by modality (acc filled in later by gpt-5.4-mini eval; here only generation stats)
def agg_of(rs):
    return {"n": len(rs),
            "doc_hit": round(sum(bool(r["hits"]["doc_hit"]) for r in rs) / max(1, len(rs)), 3),
            "avg_calls": round(sum(r["n_calls"] for r in rs) / max(1, len(rs)), 2)}
agg = {m: agg_of([r for r in results if r["modality"] == m]) for m in sorted(set(r["modality"] for r in results))}
agg["ALL"] = agg_of(results)
print("=== %s ===\n%s\nspent=$%.2f" % (args.config, json.dumps(agg, indent=1), gem.spent()), flush=True)
json.dump({"config": args.config, "agg": agg, "spent_usd": round(gem.spent(), 2),
           "tok": gem.TOK, "results": results}, open(out, "w"), indent=1)
print("TOK=%s" % gem.TOK, flush=True)
