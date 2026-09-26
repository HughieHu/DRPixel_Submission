"""Health-check one result file the moment it is written, and say so in the job log.

Every expensive mistake in this project looked fine in `sacct` and fine in the result file.
The four numbers below are the ones that would have caught them:

  empty%          server died mid-run (SIGBUS) or context overflow -> HTTP 400 -> blank answers
  search-error%   corpus unreadable from this node                 (>5% = the run is void)
  has_image%      hits with no page image: the silent-hollow case  (voided the GPT/Gemini baselines)
  n               short runs that `sacct` still reports COMPLETED

Exit code is 1 when a threshold is breached, so the sbatch can refuse to look successful.
"""
import json, os, sys

TH_ERR, TH_EMPTY, TH_IMG = 0.05, 0.05, 0.90

def load(p):
    d = json.load(open(p))
    return d if isinstance(d, list) else (d.get("results") or list(d.values()))

def main(paths):
    bad = 0
    for p in paths:
        if not os.path.exists(p):
            print("POSTCHECK %s -> MISSING" % p); bad = 1; continue
        r = [x for x in load(p) if isinstance(x, dict)]
        n = len(r)
        if not n:
            print("POSTCHECK %s -> EMPTY FILE" % os.path.basename(p)); bad = 1; continue
        empty = sum(1 for x in r if not (x.get("answer") or "").strip())
        se = [s for x in r for s in (x.get("searches") or []) if s.get("error")]
        st = sum(len(x.get("searches") or []) for x in r)
        hits = [h for x in r for s in (x.get("searches") or []) for h in (s.get("hits") or [])]
        # has_image must be computed over the IMAGE-BEARING tools only. Pooling text and figure
        # hits made this ratio ~3% for every txtfig run by construction, so it fired a false alarm
        # on a healthy run -- and, far worse, could never have fired on a genuinely hollow figure
        # channel, because text hits alone keep the pooled ratio pinned near zero either way.
        _IMG_TOOLS = ("search_figure", "search_page", "visit")
        ihits = [h for x in r for sx in (x.get("searches") or [])
                 if (sx.get("tool") or "") in _IMG_TOOLS for h in (sx.get("hits") or [])]
        if ihits:
            hits = ihits
        img = sum(1 for h in hits if h.get("has_image"))
        er = len(se) / st if st else 0.0
        em = empty / n
        ir = img / len(hits) if hits else 1.0
        flags = []
        if er > TH_ERR:   flags.append("SEARCH-ERROR %.1f%%" % (100 * er))
        if em > TH_EMPTY: flags.append("EMPTY %.1f%%" % (100 * em))
        if ir < TH_IMG:   flags.append("HAS_IMAGE only %.1f%%" % (100 * ir))
        print("POSTCHECK %-52s n=%-4d empty=%4.1f%% search_err=%4.1f%% has_image=%5.1f%%  %s"
              % (os.path.basename(p), n, 100 * em, 100 * er, 100 * ir,
                 ("<<< " + " | ".join(flags)) if flags else "ok"))
        if se[:1]:
            print("   first search error: %s" % str(se[0].get("error"))[:160])
        if flags:
            bad = 1
    return bad

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
