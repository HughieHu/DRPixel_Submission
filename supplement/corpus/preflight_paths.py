"""Abort a job BEFORE it serves, if this node cannot actually read the corpus.

The failure this exists for is silent: corpus.page_image() answers a missing PDF with a
blank white page, so a mis-rooted job keeps RUNNING, writes a normal-looking result file,
and is graded as if the model had simply failed. It cost 15 poisoned rollouts once and a
whole js-pool eval (94% search failure) another time.

Run it in the sbatch right after VLDR_ROOT is set:

    python preflight_paths.py || exit 1

Checks, in order of how early they fail:
  1. ROOT exists and holds the trees corpus.py derives (raw / corpus_cs / experiment)
  2. doc_map.json loads, and its stored paths normalise onto THIS ROOT
  3. a random sample of docs resolves to files that exist  (--min, default 0.99)
  4. one page actually renders through fitz, and is not the blank fallback
"""
import argparse, json, os, random, sys

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300, help="docs to sample")
ap.add_argument("--min", type=float, default=0.99, help="required resolve rate")
ap.add_argument("--bench", action="store_true", help="also check the new3 bench corpora")
a = ap.parse_args()

ROOT = os.environ.get("VLDR_ROOT")
if not ROOT:
    sys.exit("PREFLIGHT FAIL: VLDR_ROOT is not set")
print("PREFLIGHT node=%s ROOT=%s" % (os.uname().nodename, ROOT))

for sub in ("raw", "corpus_cs", "experiment"):
    print("  %-12s %s" % (sub, "ok" if os.path.isdir(ROOT + "/" + sub) else "MISSING"))
if not os.path.isdir(ROOT):
    sys.exit("PREFLIGHT FAIL: ROOT does not exist from this node")

if a.bench:
    os.environ["VLDR_BENCH"] = "1"
sys.path.insert(0, ROOT + "/experiment")
# a corpus.py sitting next to this file wins, so a patch can be proven from a staging
# directory before it is installed over the copy that running jobs import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus                                   # noqa: E402  (needs VLDR_ROOT first)

m = corpus._doc_map()
print("  doc_map      %d docs" % len(m))
if not m:
    sys.exit("PREFLIGHT FAIL: doc_map is empty")

# what prefix did the file actually store? purely informational, but it is the single
# most useful line in the log when a node-mismatch is suspected
sample_path = next((v.get("cl") for v in m.values() if isinstance(v.get("cl"), str)), "")
stored = next((p for p in corpus._KNOWN_ROOTS if sample_path.startswith(p)), "(unrecognised)")
print("  stored under %s   -> normalised to %s" % (stored, ROOT))

random.seed(0)
docs = random.sample(list(m), min(a.n, len(m)))
bad_cl = [d for d in docs if not (corpus.info(d) or {}).get("cl") or not os.path.exists(corpus.info(d)["cl"])]
bad_pdf = [d for d in docs if not (corpus.info(d) or {}).get("pdf") or not os.path.exists(corpus.info(d)["pdf"])]
n = len(docs)
print("  content_list %d/%d resolve (%.1f%%)" % (n - len(bad_cl), n, 100.0 * (n - len(bad_cl)) / n))
print("  origin.pdf   %d/%d resolve (%.1f%%)" % (n - len(bad_pdf), n, 100.0 * (n - len(bad_pdf)) / n))
if bad_cl[:3]:
    print("  e.g. unresolved cl : %s" % [(corpus.info(d) or {}).get("cl") for d in bad_cl[:3]])

# render one page for real -- resolving a path is not the same as being able to read it
good = next((d for d in docs if d not in bad_pdf), None)
if good is None:
    sys.exit("PREFLIGHT FAIL: no sampled doc has a readable PDF")
before = corpus.missing_report()[0]
png = corpus.page_image(good, 0)
after = corpus.missing_report()[0]
blank = after > before
print("  render       doc=%s %d bytes %s" % (good, len(png or b""), "BLANK FALLBACK" if blank else "ok"))
if blank or not png:
    sys.exit("PREFLIGHT FAIL: page render fell back to a blank image")

rate = (n - len(bad_cl)) / n
if rate < a.min:
    sys.exit("PREFLIGHT FAIL: only %.1f%% of docs resolve from this node (need %.0f%%). "
             "This node cannot read the corpus at ROOT -- do NOT serve." % (100 * rate, 100 * a.min))
print("PREFLIGHT PASS")
