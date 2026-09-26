import json, sys
src, dst, mode, limit = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
TE = "</think>"
def strip(m): return {k: v for k, v in m.items() if k in ("role","content","loss","loss_scale")}
n = 0; nsplit = 0; nmsg_in = 0; nmsg_out = 0
with open(dst, "w") as out:
    for line in open(src):
        line = line.strip()
        if not line: continue
        d = json.loads(line); n += 1
        if limit and n > limit: break
        msgs = d["messages"]; nmsg_in += len(msgs)
        new = []
        for m in msgs:
            if m.get("role")=="assistant" and ("think_scale" in m or "query_scale" in m) and isinstance(m.get("content"),str):
                k = m["content"].find(TE)
                if k != -1:
                    k += len(TE)
                    ts = 1.0 if mode=="u1" else float(m.get("think_scale",1.0))
                    qs = 1.0 if mode=="u1" else float(m.get("query_scale",1.0))
                    new.append({"role":"assistant","content":m["content"][:k],"loss_scale":ts}); nsplit += 1
                    rest = m["content"][k:]
                    if rest: new.append({"role":"assistant","content":rest,"loss_scale":qs})
                    continue
            mm = strip(m)
            if mode=="u1" and mm.get("role")=="assistant" and "loss_scale" in mm: mm["loss_scale"]=1.0
            new.append(mm)
        d["messages"] = new; nmsg_out += len(new)
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
print("  %-26s 行%d  拆分%d条  消息 %d->%d  mode=%s" % (dst.split("/")[-1], n-1 if limit and n>limit else n, nsplit, nmsg_in, nmsg_out, mode))
