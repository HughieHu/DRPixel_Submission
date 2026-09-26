# Supplementary code

The programs behind the paper: the search agent, the retrieval stack it queries, the
evaluation and grading, and the per-step counterfactual credit training recipe.

**Code only.** No corpus, no page images, no benchmark questions, no training
trajectories and no checkpoints are included here.

Every credential and private endpoint was stripped before release and replaced with an
`os.environ` lookup; see `.env.example` for the variables each component expects.
Absolute paths were replaced with `/path/to/...` placeholders.

## Layout

### `agent/` — the search agent

| file | |
|---|---|
| `react_chat.py` | the multi-turn harness. Holds the system prompt, the `<think>` / `<call_tool>` / `<answer>` protocol, the budget accounting, and the per-provider halting rules. Every result in the paper comes from this harness. |
| `agent_model.py` | provider adapters. The three backends differ more than they look: the self-hosted model emits `<think>` in-band and is halted with stop sequences; one proprietary API rejects stop sequences and is driven through native function calling; another strips the closing tag and needs its reasoning signature echoed back. |
| `configs.py` | tool sets (`page`, `text`, `text+figure`) and their retrievers. |
| `run.py` | the driver: loads a question file, runs the agent concurrently, writes one record per question. |

### `corpus/` — retrieval

| file | |
|---|---|
| `corpus.py` | document access: reading-order text, 256-**word** chunks, page rendering from the source PDF, figure crops. |
| `index.py` | vector index loading, including the packed single-file format. |
| `retriever.py` | the retrievers themselves (page-image, OCR text, text+figure). |
| `preflight_paths.py` | resolves and checks every data path before a run starts. Worth keeping: an unreachable page path does not raise, it silently returns a blank page, and the run then looks healthy while the visual channel is empty. |

### `eval/` — scoring

| file | |
|---|---|
| `judge.py` | the strict judge used for the paper's own benchmark. A directional answer that omits the gold's specific value is graded wrong. |
| `grade_vidore.py` | the looser judge used for the external document benchmark, whose long-form reference answers the strict rubric penalises systematically. The two are **not** comparable in absolute terms; each is only valid within its own benchmark. |
| `postcheck.py` | run health checks (empty answers, resolution fingerprint, retrieval errors). |

### `train/` — per-step counterfactual credit

The pipeline runs in this order:

| step | file | |
|---|---|---|
| 1 | `make_weighted_v3.py` | turn raw per-step credits into rank-based weights, one channel at a time, ranked against a pool pooled over the whole training set. |
| 2 | `mksplit.py` | split each assistant message at `</think>` so reasoning and the search query carry separate weights. |
| 3 | `compute_cfw_nextact.py` | the mediated credit: how much a reasoning step raises the likelihood of the query it actually produced. Ablation replaces the reasoning with an empty block rather than deleting it, so both conditions keep the same token structure. |
| 4 | `build_splitmed.py` | apply the mediation gate and the reasoning/query coupling, and write the training file. `build_splitmed_any.py` is the same recipe with its paths lifted out so other data slices can use it. |
| — | `train_nt2500_SPLITMED.sbatch` | the training job. |
| — | `merge_splitmed.sbatch` | merge the adapter into the base weights. |
| — | `eval_preempt_any.sbatch` | serve the merged model and run the evaluation. |

## Reproducing

The scripts assume a Slurm cluster and a corpus laid out under `VLDR_ROOT`; they are
included as the record of what was run, not as a turnkey pipeline. The two pieces that
transfer without that environment are `agent/react_chat.py` (the protocol and halting
logic) and `train/` (the credit computation, which needs only trajectories and a frozen
scoring model).
