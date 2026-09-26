# Supplementary material

Credentials and private endpoints were removed and replaced with `os.environ` lookups; see
`.env.example`. Absolute paths were replaced with `/path/to/...` placeholders.

## `bench/`

`eval500.json` — 500 questions over scientific documents.

```
qid           an opaque identifier
question      the question
gold_answer   the reference answer
modality      text | table | figure | equation     216 / 135 / 102 / 47
doc, arxiv    the source document
page          the page the evidence is on, or null
```

## `agent/`

| file | |
|---|---|
| `react_chat.py` | the multi-turn loop: system prompt, the `<think>` / `<call_tool>` / `<answer>` protocol, budget accounting, halting. |
| `agent_model.py` | three backends — the self-hosted model, GPT-5.4 through the OpenAI API, and Gemini through Vertex. |
| `configs.py` | the three tool sets: `text_only`, `txtfig_q8`, `page_q8`. |
| `run.py` | loads a question file, runs the agent concurrently, writes one record per question. |

## `eval/`

`judge.py` — `_JUDGE` is the grading rubric, `judge_answer` the call, `retrieval_hits` the
page- and document-hit computation.

## `train/`

Per-step counterfactual credit, in order:

| step | file | |
|---|---|---|
| 1 | `make_weighted_v3.py` | rank-based weights per channel, ranked against a pool taken over the whole training set. |
| 2 | `mksplit.py` | split each assistant message at `</think>` so reasoning and query carry separate weights. |
| 3 | `compute_cfw_nextact.py` | the mediated credit: how much a reasoning step raises the likelihood of the query it wrote. |
| 4 | `build_splitmed.py` | apply the mediation gate and the reasoning/query coupling; write the training file. |

## `runs_demo_trained.json`

One evaluation run of the trained model, as a sample of the record format:

```
qid, answer, correct, tok
steps[] = {i, tool, query, in_tok, out_tok, hits[{doc, page_idx}]}
```
