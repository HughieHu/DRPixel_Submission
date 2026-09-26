"""Next-action counterfactual credit for THINK segments.

For each search-step assistant message (think + call_tool in one message):
  na_think[i] = logP(action_seg | prefix, think_i present)
              - logP(action_seg | prefix, think_i ablated)
i.e. how much this step's reasoning helps produce THIS step's query -- the
mediation-corrected credit (deletion-LOO against the gold answer lets think's
contribution leak into the query it produced; here the query IS the target).

Scoring model / data format / resume identical to compute_cfw_v3_multigpu.py.
Writes `na_think` per assistant message alongside the untouched original fields.
"""
import json, os, re, argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--img_root", default="/path/to/work/rw30_sft/images")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max_pixels", type=int, default=602112)
args = ap.parse_args()

from transformers import AutoProcessor
try: from transformers import AutoModelForImageTextToText as _AM
except Exception: from transformers import AutoModelForVision2Seq as _AM
print("loading", args.model, flush=True)
proc = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, trust_remote_code=True)
_ngpu = torch.cuda.device_count()
_dmap = "cuda" if _ngpu <= 1 else "balanced"
print("GPUs=%d device_map=%s" % (_ngpu, _dmap), flush=True)
model = _AM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=_dmap,
                            trust_remote_code=True); model.eval()

THINK_SPLIT = re.compile(r"^(<think>.*?</think>\s*)(.+)$", re.S)

def rimg(p): return p if os.path.exists(p) else os.path.join(args.img_root, os.path.basename(p))
def to_parts(content, img_iter):
    parts = []
    for seg in re.split(r"(<image>)", content):
        if seg == "<image>": parts.append({"type": "image", "image": next(img_iter)})
        elif seg: parts.append({"type": "text", "text": seg})
    return parts or [{"type": "text", "text": content}]

EMPTY_THINK = "<think>\n\n</think>\n\n"

def _base(prefix_pmsgs):
    """Generation prompt WITHOUT the auto-opened <think>: Qwen3's template ends the
    generation prompt with '<think>\n', so we strip it and supply the think block
    ourselves -- otherwise prefix and full sequence disagree by one tag."""
    g = proc.apply_chat_template(prefix_pmsgs, tokenize=False, add_generation_prompt=True)
    return g[:-len("<think>\n")] if g.endswith("<think>\n") else g

def score_seg(base, pre_text, target_text, images):
    """logP(target_text | base + pre_text), summed over target tokens. Strings are
    concatenated from one shared base so both conditions tokenize identically up to
    pre_text."""
    from PIL import Image
    ims = [Image.open(q).convert("RGB") for q in images] if images else None
    text_pref = base + pre_text
    text_full = text_pref + target_text
    enc = proc(text=[text_full], images=ims, return_tensors="pt").to(model.device)
    n_pref = proc(text=[text_pref], images=ims, return_tensors="pt").input_ids.shape[1]
    labels = enc.input_ids.clone(); labels[:, :n_pref] = -100
    n_tgt = int((labels != -100).sum().item())
    if n_tgt == 0: return None
    with torch.no_grad():
        loss = model(**enc, labels=labels).loss
    return -float(loss.item()) * n_tgt

def process(rec):
    msgs = rec["messages"]
    imgs = [rimg(p) for p in rec.get("images", [])]
    img_before = [0]
    for mm in msgs: img_before.append(img_before[-1] + mm["content"].count("<image>"))
    def pmsgs_upto(k):  # messages [0, k) as parts
        it = iter(imgs[:img_before[k]])
        return [{"role": mm["role"], "content": to_parts(mm["content"], it)} for mm in msgs[:k]]
    # SPLIT format: a step is TWO assistant messages -- a think-only message
    # ("<think>...</think>") followed by an action-only message ("<call_tool ...>"
    # or "<answer>"). At inference they are one turn: think + "\n\n" + action.
    na, kind = {}, {}
    for i, mm in enumerate(msgs):
        if mm["role"] != "assistant": continue
        c = mm["content"].strip()
        if not (c.startswith("<think>") and c.endswith("</think>")): continue
        if i + 1 >= len(msgs) or msgs[i + 1]["role"] != "assistant": continue
        act = msgs[i + 1]["content"]
        if "<call_tool" in act: k = "query"
        elif "<answer>" in act: k = "answer"
        else: continue
        base = _base(pmsgs_upto(i))
        pimgs = imgs[:img_before[i]]
        lp_with = score_seg(base, c + "\n\n", act, pimgs)
        lp_wo = score_seg(base, EMPTY_THINK, act, pimgs)
        if lp_with is not None and lp_wo is not None:
            na[i] = lp_with - lp_wo; kind[i] = k
    out_msgs = []
    for i, mm in enumerate(msgs):
        mm2 = dict(mm)
        if i in na:
            mm2["na_think"] = round(na[i], 4); mm2["na_kind"] = kind[i]
        out_msgs.append(mm2)
    rec2 = dict(rec); rec2["messages"] = out_msgs
    return rec2

recs = [json.loads(l) for l in open(args.data) if l.strip()]
recs = [r for i, r in enumerate(recs) if i % args.nshard == args.shard]
if args.limit: recs = recs[:args.limit]
print("shard %d/%d: %d records" % (args.shard, args.nshard, len(recs)), flush=True)
outf = args.out.replace(".jsonl", ".shard%d.jsonl" % args.shard); n_ok = 0
_nfile = outf + ".n"; _skip = 0
if os.path.exists(_nfile):
    try: _skip = int(open(_nfile).read().strip())
    except Exception: _skip = 0
elif os.path.exists(outf):
    _skip = sum(1 for _ in open(outf))
if _skip:
    print("RESUME: skipping %d records" % _skip, flush=True)
    recs = recs[_skip:]
with open(outf, "a") as fo:
    for j, r in enumerate(recs):
        r2 = None   # a record that raises must not leave r2 unbound: the progress
                    # print below sits OUTSIDE the try and killed two shards with
                    # NameError the moment a resumed run hit a bad first record.
        try:
            r2 = process(r)
            if r2: fo.write(json.dumps(r2) + "\n"); fo.flush(); n_ok += 1
        except Exception as e:
            print("  rec %d ERR: %s" % (j, str(e)[:140]), flush=True)
            # Remember it: the counter below advances regardless, so without this the
            # record is skipped for good and the arm can never reach full coverage.
            open(outf + ".failed", "a").write("%d\n" % (_skip + j))
            try:
                import torch as _t; _t.cuda.empty_cache()
            except Exception:
                pass
        open(_nfile, "w").write(str(_skip + j + 1))
        if j % 20 == 19:
            try:
                import torch as _t; _t.cuda.empty_cache()
            except Exception:
                pass
        if j < 3 or j % 25 == 0:
            t = [mm.get("na_think") for mm in (r2["messages"] if r2 else []) if "na_think" in mm]
            print("  [%d/%d] na_think=%s" % (_skip + j, _skip + len(recs), t), flush=True)
print("DONE shard %d: %d ok" % (args.shard, n_ok), flush=True)
