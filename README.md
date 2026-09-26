# DR Pixel

Visual deep research over scientific documents. An agent issues searches against a
corpus of rendered pages, reads the page images it retrieves, and answers questions
whose evidence lives in figures, tables and equations rather than in running text.

This repository accompanies the paper (under review).

## Contents

[`supplement/`](supplement/) — the programs behind the paper. Code only: no corpus, no
page images, no benchmark questions, no training trajectories, no checkpoints.

| path | |
|---|---|
| [`supplement/agent/`](supplement/agent/) | the search agent: harness, provider adapters, tool configuration, run driver |
| [`supplement/corpus/`](supplement/corpus/) | retrieval: document access, index loading, the three retrievers, path preflight |
| [`supplement/eval/`](supplement/eval/) | the two judges and the run health checks |
| [`supplement/train/`](supplement/train/) | per-step counterfactual credit, the mediation gate, and the training/merge/eval jobs |

See [`supplement/README.md`](supplement/README.md) for what each file does and
[`supplement/.env.example`](supplement/.env.example) for the configuration each component
expects. All credentials and private endpoints were removed before release.

## Still to come

The benchmark (questions, reference answers, page-level evidence annotations), the corpus
build for the 99,055-document / 1,062,153-page collection, and model weights.
