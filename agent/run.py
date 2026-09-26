import argparse, json, os, time
import configs, retriever, react, judge
import os as _os
HARNESS = _os.environ.get("HARNESS", "react")
if HARNESS == "chat":
    import react_chat as _rc

import agent_model as gem

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, choices=list(configs.CONFIGS))
_VLDR = os.environ.get("VLDR_ROOT", "/path/to/vldr")
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
_model_safe = _model.replace("/", "_")
out = args.out or _VLDR + "/experiment/results/%s__%s.json" % (_model_safe, args.config)
os.makedirs(os.path.dirname(out), exist_ok=True)
print("AGENT_MODEL=%s | out=%s" % (_model, out), flush=True)

done = {}
if os.path.exists(out):
    try:
        prev = json.load(open(out))
        for r in (prev.get("results", prev) if isinstance(prev, dict) else prev):

            if r.get("failed") or not (r.get("answer") or "").strip() or \
               (r.get("answer") or "").startswith("ERROR"):
                continue
            done[r["_qid"]] = r
    except Exception:
        pass
print("config=%s tools=%s | %d questions (%d already done)" % (args.config, cfg, len(bench), len(done)), flush=True)

print("building retrievers...", flush=True)
tools = {tn: retriever.make(par, args.device) for tn, par in cfg.items()}

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
CONC = int(os.environ.get("CONCURRENCY", "4"))
results = list(done.values())
todo = [it for it in bench if it["_qid"] not in done]
t0 = time.time(); _lock = threading.Lock(); _n = [0]
print("processing %d questions, CONCURRENCY=%d" % (len(todo), CONC), flush=True)

_extra = react.PROMPT_EXTRA.get(args.config, "")
_rewrite = gem.rewrite_query if args.config in getattr(configs, "REWRITE_CONFIGS", set()) else None
if _rewrite:
    print("QUERY REWRITE ON: separate %s rewrites each search_page query -> ~%sw NL query"
          % (gem.REWRITE_MODEL, gem.REWRITE_WORDS), flush=True)
def _process(item):
    _fn = _rc.run_react_chat if HARNESS == "chat" else react.run_react
    _kw = {"anchor_img": item.get("anchor_img")} if HARNESS == "chat" else {}
    res = _fn(item["question"], tools, budget=args.budget, max_steps=args.budget + 6, extra=_extra, rewrite=_rewrite, **_kw)
    return {"_qid": item["_qid"], "modality": item["modality"], "answer_type": item["answer_type"],
            "doc": item["doc"],
            "question": item.get("question"), "gold_answer": item.get("gold_answer"),
            "answer": res["answer"], "failed": res.get("failed"), "correct": None,
            "reason": "pending gpt-5.4-mini eval",

            "n_calls": res["n_calls"], "calls_by_tool": res.get("calls_by_tool"),
            "in_tok": res.get("in_tok"), "out_tok": res.get("out_tok"), "reasoning_tok": res.get("reasoning_tok"),
            "wall_s": res.get("wall_s"),

            "hits": judge.retrieval_hits(item, res["searches"]), "searches": res["searches"],
            "steps": res.get("steps"), "meta": res.get("meta")}

with ThreadPoolExecutor(max_workers=CONC) as ex:
    futs = [ex.submit(_process, it) for it in todo]
    for fut in as_completed(futs):
        rec = fut.result()
        with _lock:
            results.append(rec); _n[0] += 1
            if _n[0] % 5 == 0 or _n[0] == len(todo):
                json.dump({"config": args.config, "results": results}, open(out, "w"), indent=1)
                print("  %d/%d | $%.2f | 429=%d | %.0fs" % (_n[0], len(todo), gem.spent(),
                                                            gem.TOK.get("429", 0), time.time() - t0), flush=True)

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
