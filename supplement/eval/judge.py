"""Scoring. (1) judge_answer: gemini-3.5-flash grades the agent answer vs gold by answer_type
(verdict/comparison/trend/regime), lenient -- same conclusion/direction matches, paraphrase OK
(vjudge style). (2) retrieval_hits: rule-based from the search trace -- did any search surface the
gold doc / exact block (rt.hash, figure) / gold page (rt.page)? For text-tool table questions the
useful signal is doc_hit + the gold info appearing in a retrieved passage (judged via the answer
grade, since the agent answered from those passages)."""
import json
import gem

_JUDGE = """You STRICTLY grade an answer to a question about a research paper's figure/table/finding.
The GOLD states a SPECIFIC finding one can only know by reading the figure/table -- often exact
numbers, named models/methods/datasets, or a precise quantitative relationship.

GRADING RULE:
1. If the GOLD contains specific numbers, named entities (models/methods/datasets), or a precise
   magnitude, mark correct=true ONLY IF the PREDICTED answer states that SAME specific finding:
   the same numbers (within rounding) AND/OR the same named entities AND/OR the same precise
   relationship. Otherwise correct=FALSE. In particular:
   - Generic / directional answer that gets the trend right but OMITS the gold's specific numbers => WRONG.
   - Answer citing DIFFERENT numbers or DIFFERENT models/methods than the gold => WRONG (it guessed
     from prior knowledge, did not read the source).
   - Vague hedging ("levels off", "tends to", "generally", "competitive") in place of the gold's
     specific finding => WRONG.
2. If the GOLD has NO specific numbers/entities (a pure yes/no verdict or a purely qualitative
   direction), then a correctly matching verdict/direction is enough.
Paraphrase of the SAME specific finding is fine; extra correct detail is fine; a contradicted or
reversed direction is always WRONG.

answer_type = {atype} (verdict / comparison / trend / regime).

CALIBRATION EXAMPLES:
[CORRECT] GOLD: "No, it stays stable..." (no numbers) | PRED: "No, ... remains bounded, avoids collapse" -> verdict matches, gold has no specifics.
[CORRECT] GOLD: "operational factors; limiters = latent size and number of modules" | PRED: "operational overhead; size and number of processed tensors" -> states the same specific finding.
[WRONG] GOLD: "Jina-ColBERT-v2 within ~0.4 nDCG@10 of mDPR-FT on MIRACL" | PRED: "a single model such as CORA or mAggretriever is competitive" -> different models, none of the gold's specifics.
[WRONG] GOLD: "EMOVA 72B trails on RealWorldQA 71.0 vs 75.4" | PRED: "InternVL3-78B, Qwen-VL2.5-72B trail GPT-4o" -> different models, missing 71.0/75.4.
[WRONG] GOLD: "edge vanishes: 66.6 vs 62.7 (Qwen2.5) but 70.3 vs 70.9 (Qwen3)" | PRED: "the advantage diminishes" -> correct direction but NONE of the gold's specific numbers.
[WRONG] GOLD: "rises 2->4, peaks ~4, dips at 5" | PRED: "levels off / plateaus" -> vague, misses the specific peak/dip.

QUESTION: {q}
GOLD ANSWER: {gold}
PREDICTED ANSWER: {pred}
First state the gold's KEY specific finding, then decide.
Output ONLY: {{"gold_key_finding": "<the specific finding the gold makes>", "correct": true or false, "reason": "<one sentence: does PRED state it?>"}}"""


def judge_answer(question, gold, pred, answer_type):
    pred = (pred or "").strip()
    if not pred or pred.startswith("ERROR:"):
        return {"correct": False, "reason": "no answer / error"}
    try:
        r, _ = gem.jcall([gem.text_part(_JUDGE.format(atype=answer_type, q=question,
                                                       gold=gold, pred=pred))])
        return {"correct": bool(r.get("correct")), "reason": str(r.get("reason", ""))[:200]}
    except Exception as e:
        return {"correct": False, "reason": "judge error: " + str(e)[:120]}


def retrieval_hits(item, searches):
    """item = bench entry (has 'doc' + 'rt'); searches = react trace. Returns hit flags."""
    rt_doc = item["doc"]
    rt = item.get("rt", {})
    rt_hash = rt.get("hash")
    rt_page = rt.get("page")
    all_hits = [h for s in searches for h in s.get("hits", [])]
    doc_hit = any(h.get("doc") == rt_doc for h in all_hits)
    block_hit = (any(h.get("doc") == rt_doc and (h.get("hash") or "").split(".")[0] == str(rt_hash).split(".")[0]
                     for h in all_hits) if rt_hash else None)
    page_hit = (any(h.get("doc") == rt_doc and h.get("page_idx") == rt_page
                    for h in all_hits) if rt_page is not None else None)
    return {"doc_hit": doc_hit, "block_hit": block_hit, "page_hit": page_hit,
            "n_searches": len(searches)}
