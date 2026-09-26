import argparse
import collections
import glob
import hashlib
import json
import os
import statistics as st
import sys

OLD, NEW = "/path/to/work/", "/path/to/work/"

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True, help="mksplit output for this arm")
ap.add_argument("--na", required=True, nargs="+", help="next-action shard globs")
ap.add_argument("--out_prefix", required=True, help="writes <prefix>_thid.jsonl and _net.jsonl")
ap.add_argument("--allow-missing", action="store_true",
                help="build even if some trajectories have no next-action credit")
args = ap.parse_args()

def key(r):
    return hashlib.sha1(json.dumps([m.get("content") for m in r["messages"]],
                                   ensure_ascii=False).encode()).hexdigest()

def is_think(m):
    c = (m.get("content") or "").strip()
    return m["role"] == "assistant" and c.startswith("<think>") and c.endswith("</think>")

def kept(m):
    return (m.get("loss_scale") or 0) > 0 and m.get("loss") is not False

na = {}
files = [f for g in args.na for f in sorted(glob.glob(g))]
for f in files:
    for l in open(f):
        r = json.loads(l)
        na[key(r)] = {i: m["na_think"] for i, m in enumerate(r["messages"]) if "na_think" in m}
print("read %d shard(s), %d trajectories carry a next-action credit" % (len(files), len(na)))

rows = [json.loads(l) for l in open(args.split)]
hit = [r for r in rows if key(r) in na]
cov = 100.0 * len(hit) / len(rows)
print("this arm has %d trajectories, %d matched (%.1f%%)" % (len(rows), len(hit), cov))
if len(hit) < len(rows) and not args.allow_missing:
    sys.exit("coverage %.1f%%, %d trajectories missing. The threshold is the median of THIS arm's distribution, so it cannot be built from a partial set.\n"
             "   Complete the coverage and rerun, or pass --allow-missing deliberately." % (cov, len(rows) - len(hit)))

allna = [v for r in hit for v in na[key(r)].values()]
if not allna:
    sys.exit("no next-action credits at all; cannot set a threshold")
thr = st.median(allna)
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
            m["loss_scale"] = w_rescue
            m.pop("loss", None)
            c["rescued"] += 1
        elif (not keep) and was:
            m["loss_scale"] = 0.0
            c["dropped"] += 1
        else:
            c["unchanged_" + ("kept" if keep else "masked")] += 1

thid_p, net_p = args.out_prefix + "_thid.jsonl", args.out_prefix + "_net.jsonl"
with open(thid_p, "w") as fo:
    for r in rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")

seen, missing, n_rw = set(), [], 0
net_rows = []
for r in rows:
    r2 = dict(r)
    imgs = r.get("images") or []
    if imgs:
        out = []
        for p in imgs:
            if p.startswith(OLD):
                p = NEW + p[len(OLD):]
                n_rw += 1
            out.append(p)
            if p not in seen:
                seen.add(p)
                if not os.path.exists(p):
                    missing.append(p)
        r2["images"] = out
    net_rows.append(r2)
if missing:
    for p in missing[:5]:
        print("   missing image: %s" % p)
    sys.exit("%d images in the _net variant do not resolve; refusing to write. An unreachable image does not raise, it feeds a blank page." % len(missing))
with open(net_p, "w") as fo:
    for r in net_rows:
        fo.write(json.dumps(r, ensure_ascii=False) + "\n")

tk = [m for r in rows for m in r["messages"] if is_think(m)]
nz = [m["loss_scale"] for r in rows for m in r["messages"]
      if m["role"] == "assistant" and kept(m)]
print("na values=%d  median threshold=%.3f  rescue weight=%.4f" % (len(allna), thr, w_rescue))
print(dict(c))
print("think mask rate -> %.1f%%" % (sum(1 for m in tk if not kept(m)) / len(tk) * 100))
print("nonzero assistant weights: n=%d mean=%.4f" % (len(nz), st.mean(nz)))
print("rewrote %d image paths; all %d distinct images resolve" % (n_rw, len(seen)))
print("wrote %s\n      %s" % (thid_p, net_p))
