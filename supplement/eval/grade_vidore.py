"""Grade ViDoRe V3 agent runs with the canonical judge, and record HOW they were graded.

The first ViDoRe numbers (flash 60.6 / eck150 59.0 / base 54.7) came from a heredoc that was never
saved, and the graded files carry no judge metadata -- so a later run graded by any other script is
not comparable to them. This script exists so that never happens again: it writes `judge_meta` into
every file it touches, and the whole set is meant to be re-graded with it in one pass.

Judge: judge.py::_JUDGE, gpt-5.4-mini, 3x majority, temperature fixed by the endpoint default.
Endpoints: the two Azure deployments from grade_airqa_perrec.py, probed live and pruned before use
(a dead endpoint previously graded only 81% of a file while still exiting 0).
"""
import os, sys, json, re, urllib.request, ssl, time, threading, hashlib, itertools
from concurrent.futures import ThreadPoolExecutor

S = "/path/to/work"
E = S + "/VL_Deep_Research/experiment"
sys.path.insert(0, E)
# Both Azure deployments are down (artur 401 since 8/28, kartrina as of 9/6), so grading now goes
# through the official OpenAI endpoint. Keeping the list shape means the round-robin / prune / retry
# machinery below is unchanged; there is simply one entry.
_EPS = [("https://api.openai.com/v1/responses", open(S + "/.openai_official_key").read().strip())]
_rr = itertools.count()
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
TOK = {"in": 0, "out": 0, "calls": 0, "fail": 0}; _lk = threading.Lock()

def _resp(prompt):
    body = json.dumps({"model": "gpt-5.4-mini", "input": [{"role": "user", "content": prompt}],
                       "max_output_tokens": 400, "reasoning": {"effort": "low"}}).encode()
    delay = 2.0
    for attempt in range(8):
        url, key = _EPS[(next(_rr) + attempt) % len(_EPS)]
        try:
            req = urllib.request.Request(url, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=120, context=CTX)); u = d.get("usage") or {}
            with _lk:
                TOK["in"] += u.get("input_tokens", 0) or 0; TOK["out"] += u.get("output_tokens", 0) or 0; TOK["calls"] += 1
            return "".join(c.get("text") or "" for it in d.get("output", []) for c in (it.get("content") or []))
        except Exception:
            time.sleep(delay); delay = min(delay * 1.8, 60)
    with _lk: TOK["fail"] += 1
    return ""

def _prune_dead_endpoints():
    live = []
    for url, key in _EPS:
        try:
            b = json.dumps({"model": "gpt-5.4-mini", "input": [{"role": "user", "content": "ok"}],
                            "max_output_tokens": 16, "reasoning": {"effort": "low"}}).encode()
            urllib.request.urlopen(urllib.request.Request(url, data=b,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=45, context=CTX)
            live.append((url, key)); print("judge endpoint OK:", url.split("//")[1].split(".")[0], flush=True)
        except Exception as e:
            print("judge endpoint DEAD (dropped):", url.split("//")[1].split(".")[0], "--", str(e)[:70], flush=True)
    if not live: raise SystemExit("FATAL: no live judge endpoint -- refusing to run")
    _EPS[:] = live

VIDORE_JUDGE = """You grade an answer to a question about a document (industrial manual, report,
infographic, regulation). The GOLD is a reference answer written from the source document.

Mark correct=true if the PREDICTED answer conveys the SAME substance as the GOLD: the same entities,
steps, values, or conclusion the gold states. Paraphrase is fine. Extra correct detail is fine. A
different ordering of the same steps is fine.
Mark correct=false if the prediction contradicts the gold, names different entities/values where the
gold is specific, answers a different question, is too vague to show the source was read, or is empty.
If the gold is a multi-part answer, the prediction must cover its main parts, not just one.

QUESTION: {q}
GOLD ANSWER: {gold}
PREDICTED ANSWER: {pred}

Output ONLY: {{"correct": true or false, "reason": "<one sentence>"}}"""
JUDGE_MD5 = hashlib.md5(VIDORE_JUDGE.encode()).hexdigest()[:12]

def _verdict(q, gold, ans):
    """3 votes, majority. A vote that fails to return parseable JSON is dropped, not counted as
    wrong -- an endpoint hiccup must not masquerade as an incorrect answer."""
    if not (ans or "").strip() or str(ans).startswith("ERROR:"):
        return False
    p = VIDORE_JUDGE.format(q=str(q)[:600], gold=str(gold)[:2000], pred=str(ans)[:3000])
    # Only one of the two Azure deployments is alive (artur has been 401 since 8/28), so 16 threads
    # x 3 votes rate-limit it into empty responses. Keep asking until 3 usable votes land, with a
    # hard ceiling -- an unparseable reply must never be silently scored as "incorrect".
    votes = []
    for _ in range(9):
        if len(votes) >= 3: break
        t = _resp(p).strip()
        m = re.search(r'\{.*\}', t, re.S)
        if not m: continue
        try:
            votes.append(bool(json.loads(m.group(0)).get("correct")))
            continue
        except Exception:
            pass
        # json.loads chokes on a LaTeX-bearing reason ("\\frac" is an invalid JSON escape), which
        # made every physics/equation question fail all nine votes and silently drop out of the
        # denominator. The verdict itself is a plain boolean -- read it directly.
        mv = re.search(r'"correct"\s*:\s*(true|false)', m.group(0), re.I)
        if mv: votes.append(mv.group(1).lower() == "true")
    if not votes: return None          # deferred to the retry pass, never scored as incorrect
    return sum(votes) * 2 >= len(votes) + (len(votes) % 2 == 0)

def grade(path, workers=int(os.environ.get('GRADE_WORKERS','4'))):
    o = json.load(open(path)); rs = o["results"]
    todo = [r for r in rs if r.get("correct") is None or FORCE]
    print("%s: %d records, grading %d" % (os.path.basename(path), len(rs), len(todo)), flush=True)
    def one(r):
        r["correct"] = _verdict(r.get("question", ""), r.get("gold_answer", ""), r.get("answer", ""))
    def sweep(items, workers, tag):
        with ThreadPoolExecutor(workers) as ex:
            for i, _ in enumerate(ex.map(one, items), 1):
                if i % 250 == 0:
                    print("   %s %d/%d  calls=%d fail=%d" % (tag, i, len(items), TOK["calls"], TOK["fail"]), flush=True)
    sweep(todo, workers, "pass1")
    for rnd in range(3):
        left = [r for r in todo if r.get("correct") is None]
        if not left: break
        print("   retry %d: %d records the judge could not score, at 2 workers" % (rnd + 1, len(left)), flush=True)
        time.sleep(120)                 # let the rate-limit window clear before trying again
        sweep(left, 2, "retry%d" % (rnd + 1))
    stuck = [r for r in rs if r.get("correct") is None]
    # Three retry rounds failing on the same records means a deterministic refusal (Azure content
    # filtering), not a rate-limit window -- more retries cannot help. A handful of those must not
    # destroy a two-hour grading run, but they must also never be silently scored as incorrect:
    # leave them unscored, drop them from the denominator, and say so in judge_meta.
    if len(stuck) > 0.01 * len(rs):
        raise SystemExit("FATAL: %s left %d/%d records ungraded -- that is an outage, not a few "
                         "refusals; refusing to report a partially graded file"
                         % (os.path.basename(path), len(stuck), len(rs)))
    if stuck:
        print("   %d records the judge would not score, excluded from the denominator: %s"
              % (len(stuck), ", ".join(str(r.get("_qid")) for r in stuck[:10])), flush=True)
    ungraded = len(stuck)
    o["judge_meta"] = {"judge": "grade_vidore.py::VIDORE_JUDGE", "judge_md5": JUDGE_MD5, "model": "gpt-5.4-mini",
                       "votes": 3, "grader": "grade_vidore.py", "n_scored": len(rs) - ungraded,
                       "n_unscored": ungraded, "unscored_qids": [r.get("_qid") for r in stuck],
                       "graded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = path + ".tmp"; json.dump(o, open(tmp, "w")); os.replace(tmp, path)
    scored = [r for r in rs if r.get("correct") is not None]
    n = len(scored); ok = sum(1 for r in scored if r["correct"]); dh = sum(1 for r in scored if r.get("gold_hit"))
    hit_ok = sum(1 for r in scored if r["correct"] and r.get("gold_hit"))
    print("  -> Acc %.1f  DocHit %.1f  Acc|DH %.1f" % (100*ok/n, 100*dh/n, 100*hit_ok/max(dh,1)), flush=True)
    return {"file": os.path.basename(path), "n": n, "acc": 100*ok/n, "dh": 100*dh/n, "acc_dh": 100*hit_ok/max(dh,1)}

if __name__ == "__main__":
    FORCE = "--force" in sys.argv
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    _prune_dead_endpoints()
    print("judge md5 %s  force=%s" % (JUDGE_MD5, FORCE), flush=True)
    out = [grade(f) for f in files]
    print("\n%-44s %6s %6s %6s %7s" % ("file", "n", "Acc", "DocHit", "Acc|DH"))
    for r in out: print("%-44s %6d %6.1f %6.1f %7.1f" % (r["file"], r["n"], r["acc"], r["dh"], r["acc_dh"]))
    print("\ncalls=%d fail=%d in_tok=%d out_tok=%d" % (TOK["calls"], TOK["fail"], TOK["in"], TOK["out"]))
