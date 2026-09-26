import json, glob, hashlib, sys, statistics as st, collections

S = "/path/to/work"
out = sys.argv[1] if len(sys.argv) > 1 else S + "/sft_train_nt2500_SPLITMED.jsonl"

def key(r):
    return hashlib.sha1(json.dumps([m.get("content") for m in r["messages"]],
                                   ensure_ascii=False).encode()).hexdigest()

def is_think(m):
    c = (m.get("content") or "").strip()
    return m["role"] == "assistant" and c.startswith("<think>") and c.endswith("</think>")

def kept(m):
    return (m.get("loss_scale") or 0) > 0 and m.get("loss") is not False

na = {}
for f in sorted(glob.glob(S + "/cfw_nextact_nt2500.shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        na[key(r)] = {i: m["na_think"] for i, m in enumerate(r["messages"]) if "na_think" in m}
allna = [v for d in na.values() for v in d.values()]
thr = st.median(allna)

rows = [json.loads(l) for l in open(S + "/sft_train_nt2500_SPLIT.jsonl")]
w_rescue = st.median([m["loss_scale"] for r in rows for m in r["messages"]
                      if is_think(m) and kept(m)])

c = collections.Counter()
for r in rows:
    d = na.get(key(r))
    if d is None:
        c["records_without_na(native kept)"] += 1
        continue
    msgs = r["messages"]
    for i, m in enumerate(msgs):
        if not is_think(m) or i not in d:
            continue
        nq = msgs[i + 1] if i + 1 < len(msgs) else {}
        keep = d[i] >= thr and kept(nq)
        was = kept(m)
        if keep and not was:
            m["loss_scale"] = w_rescue; m.pop("loss", None); c["rescued"] += 1
        elif (not keep) and was:
            m["loss_scale"] = 0.0; c["dropped"] += 1
        else:
            c["unchanged_" + ("kept" if keep else "masked")] += 1

with open(out, "w") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")

tk = [m for r in rows for m in r["messages"] if is_think(m)]
print("na records=%d  na values=%d  median threshold=%.3f  rescue weight=%.4f"
      % (len(na), len(allna), thr, w_rescue))
print(dict(c))
print("think mask rate: SPLITW 53.5%% -> SPLITMED %.1f%%"
      % (sum(1 for m in tk if not kept(m)) / len(tk) * 100))
nz = [m["loss_scale"] for r in rows for m in r["messages"]
      if m["role"] == "assistant" and kept(m)]
print("nonzero assistant weights: n=%d mean=%.4f" % (len(nz), st.mean(nz)))
print("wrote", out)
