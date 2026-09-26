import json, bisect, glob

SHARDS = sorted(glob.glob("/path/to/vldr/cfw_v3_full.shard*.jsonl"))
OUT    = "/path/to/work/sft_train_cfwV3.jsonl"
Q_THR, T_THR, MAXW = 0.0, 0.0, 2.0

recs = []
for sh in SHARDS:
    recs += [json.loads(l) for l in open(sh) if l.strip()]
print(f"loaded {len(recs)} records from {len(SHARDS)} shards")

q_all = [m["cf_query"] for r in recs for m in r["messages"] if "cf_query" in m]
t_all = [m["cf_think"] for r in recs for m in r["messages"] if "cf_think" in m]
pos_q = sorted(c for c in q_all if c > Q_THR)
pos_t = sorted(c for c in t_all if c > T_THR)
print(f"cf_query: n={len(q_all)}  >{Q_THR}: {len(pos_q)} ({100*len(pos_q)/max(1,len(q_all)):.1f}% kept)  "
      f"range [{min(q_all):.2f},{max(q_all):.2f}]")
print(f"cf_think: n={len(t_all)}  >{T_THR}: {len(pos_t)} ({100*len(pos_t)/max(1,len(t_all)):.1f}% kept)  "
      f"range [{min(t_all):.2f},{max(t_all):.2f}]")

def rank01(x, pool): return bisect.bisect_right(pool, x) / len(pool) if pool else 0.0
def qmap(cf): return 0.0 if cf <= Q_THR else round(MAXW * rank01(cf, pos_q), 4)
def tmap(cf): return 0.0 if cf <= T_THR else round(MAXW * rank01(cf, pos_t), 4)

nq = nq_mask = nt_mask = 0
sq = st = 0.0
for r in recs:
    for m in r["messages"]:
        if m.get("role") != "assistant": continue
        if "cf_query" in m or "cf_think" in m:
            cfq = m.pop("cf_query", None); cft = m.pop("cf_think", None)
            qs = qmap(cfq) if cfq is not None else 1.0
            ts = tmap(cft) if cft is not None else 1.0
            m["query_scale"] = qs; m["think_scale"] = ts
            nq += 1; sq += qs; st += ts
            nq_mask += (qs == 0.0); nt_mask += (ts == 0.0)

with open(OUT, "w") as f:
    for r in recs:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"\nwrote {OUT}  ({len(recs)} recs)")
print(f"search-step msgs weighted: {nq}")
print(f"  query_scale: mean {sq/max(1,nq):.3f}  masked {nq_mask} ({100*nq_mask/max(1,nq):.1f}%)")
print(f"  think_scale: mean {st/max(1,nq):.3f}  masked {nt_mask} ({100*nt_mask/max(1,nq):.1f}%)")
