# Code Guide

## Project Snapshot

This directory contains the experimental harness for the paper draft
`ksubconsitent`.

Current scope:

- `STAMP`, the active one-consistent Stable Topic Assignment via Marginal
  Potentials algorithm. The command-line route remains `ops`, and current CSV
  output still labels it as `OPS`, for compatibility with existing HPC runs.
- `GongD1`, a one-pass Gong--Liu--Fang-style non-monotone baseline
  specialized to one total capacity dimension and unit costs
- `Pham22`, a one-pass Pham et al.-style budgeted non-monotone baseline
- main diffusion model: KIC with theorem-aligned suitability-adjusted profit
- optional robustness diffusion model: LT for Email/Facebook only, using the
  same suitability-adjusted profit

Current implementation status:

- prefix-stream plumbing is in place
- objective evaluation is prefix-aware via `f_t` on `V_t`
- active profit value is `expected topic-union spread + mu * suitability_sum`;
  the old unit-cost objective `spread - support_size` is no longer active
- `main.cpp` exposes only the active `ops`, `gong`, and `pham` routes
- preprocessing now emits a shared `.bin` with `node_id == arrival order`
- the driver now accepts `B_factor`, not an absolute `B`
- each run keeps one fixed `B` and records prefix-wise statistics indexed by `t`
- reporting is sampled at 20 shared checkpoints, including `t=1` and `t=n`
- the helper `.sh` scripts take one `B_FACTOR` argument per invocation
- for `ops`, `gong`, and `pham`, the reported query count is cumulative over
  every prefix from `1` to the current `t`
- `consistency_violations` is accumulated as the excess number of newly added
  element-label pairs beyond the `1`-consistency budget, i.e.
  `max(0, new_pair_count - 1)` per reported step
- CSV route labels for the active suite are `OPS`, `GongD1`, and `Pham22`;
  interpret `OPS` as the legacy label for paper algorithm `STAMP`
- fixed experimental defaults: `alpha=1.505` for `ops`, `alpha=0.4` for
  `gong` and `pham`, `epsilon=0.1`, and in the
  `kic` benchmark decision oracle `KIC_MC=200` with `seed=42`
- KIC and LT checkpoint values are re-evaluated after the online decision phase
  with `--eval-mc` samples, defaulting to `10000`. The CSV `f_value` is the
  high-MC checkpoint estimate; `decision_f_value` preserves the low-MC estimate
  used during online decisions.
- CSV rows are appended and flushed immediately at each reported checkpoint.
  A killed job therefore preserves all completed checkpoint rows.
- `gong` and `pham` support resumable state files through `--state <path>`.
  The state is saved at each reported checkpoint and loaded automatically if
  the path already exists with matching run parameters.
- memory reporting now separates the graph-load baseline from the algorithm's
  extra peak RSS; use `algo_mem_mb` for fair algorithm-to-algorithm comparison
- KIC parallelism is OpenMP-based inside the Monte Carlo value oracle. The
  LT oracle uses the same OpenMP pattern. The route loops themselves are
  sequential, so large Slurm jobs should request one task with
  `--cpus-per-task`, export `OMP_NUM_THREADS`, or pass `--threads`.
- STAMP now caches the sorted fixed-weight vector and only recomputes the
  threshold after an accepted insertion/replacement.
- GongD1/Pham22 now reuse current-state prefix values when selecting the
  reported output and only re-evaluate stale best snapshots when needed.

## Current Semantic Model

The key semantic point for this project is prefix evaluation.

- At time `t`, only the observed prefix `V_t` is available.
- The objective is not a fixed global `f` on the full dataset.
- The correct view is a family of prefix-restricted objectives `f_t`.
- Any evaluation or marginal computation must ignore items outside `V_t`.

This matters for both the value oracle and the experiment driver:

- `evaluate(prefix_t, x)` should score only what is visible at time `t`
- `marginal(prefix_t, x, u, label)` should only use prefix-visible structure
- `OPT_t` is the optimum on `V_t`, not on the full dataset

Budget semantics are separate from prefix semantics:

- the driver reads a scalar `B_factor`
- after loading the binary graph, it computes
  `B = floor(B_factor * |V_n|)`
- `|V_n|` means the full graph size `n` from the loaded `.bin`
- `B` is then passed to the paper routes as the fixed cardinality budget
- do not recompute `B` from the current prefix `V_t`

## File Map

### Core runtime files

- `src/main.cpp`
  prefix-stream experiment driver for `ops`, `gong`, and `pham`.
- `src/prefix_stream.h`
  prefix state helper for the visible set `V_t`.
- `src/kfunctions.h`
  objective oracle interface with prefix-aware `evaluate` and `marginal`.
- `src/kfunctions_impl.h`
  compile-time binding for either `KFUNC_KIC` or `KFUNC_LT`. Revenue and sensor
  objectives are disabled for this draft.
- `src/mygraph.h`
  graph container, binary I/O, node/edge weights, and prefix-friendly metadata.
- `src/algs/result.h`
  result struct used by the paper algorithms and CSV output.
- `src/algs/stream_utils.h`
  shared helpers for label selection and prefix-suffix rebuilds.
- `src/algs/potentialswap.h`
  STAMP implementation using fixed insertion weights and the sorted-weight
  manuscript threshold.
- `src/algs/streaming_literature.h`
  one-pass non-monotone streaming baselines for GongD1 and Pham22 under the
  same prefix-restricted oracle.

### Objective implementation

- `src/objectvalue/kic.h`
- `src/objectvalue/lt.h`
- `src/objectvalue/suitability.h`
  deterministic topic-suitability generation and lookup for the active profit
  oracle
- `chart/buildActiveMetrics.py`
  plotting helper for the active four comparison axes:
  `f_value`, `consistency_violations`, `queries`, and `time_sec`.

## Input / Output Conventions

Current preprocessing assumes a binary `tinyGraph` format with:

- node count `n`
- edge count `m`
- label count `K`
- node weights / alpha values
- `part_id` metadata retained for compatibility
- edge weights

The current convention for every application is:

- node ids are assigned by first appearance in `edges.txt`
- therefore `node_id == arrival order`
- `part_id[i]` is stored as `i` in the `.bin` and is not used by the active
  paper code

The current command-line driver now uses the paper scope directly:

- `graph.bin route B_factor [options] [csv_path]`
- each binary has one compiled objective; optional matching objective tokens
  `kic` and `lt` are accepted by the corresponding binaries, while `revenue`
  and `sensor` are intentionally disabled
- active `route` values are `ops`, `gong`, and `pham`
- `B_factor` is the budget factor; internal budget is
  `B = floor(B_factor * |V_n|)` with `|V_n|` taken from the loaded full graph
- `--epsilon` controls the geometric-guess grids for `gong` and `pham`
- `--threads` controls the OpenMP thread count for Monte Carlo objective calls
  when the binary is built with `-fopenmp`.
- `--profit-mu`, `--suit-seed`, `--eta-min`, `--eta-max`, and `--xi-max`
  control the deterministic suitability-adjusted profit term. Keep
  `profit_mu * eta_max <= 1` for the nonnegativity condition used by the paper.
- `--mc`, `--kic-mc`, or `--lt-mc` controls the decision-oracle Monte Carlo
  sample count.
- `--eval-mc` controls the high-MC checkpoint reevaluation pass; set it to `0`
  to disable reevaluation.
- `--state <path>` saves/loads resumable state for `gong` and `pham`.

The intended output for experiments is a CSV row per reported checkpoint `t`.
Rows are written as soon as a checkpoint is reached, not only after the route
finishes. Duplicate checkpoint rows are skipped when graph, route, budget,
parameters, seed, and `t` already match an existing CSV row. Each row contains
at least the four main comparison axes:

- `f_value`
- `consistency_violations`
- `queries`
- `time_sec`

For KIC and LT, `f_value` is the high-MC checkpoint estimate when
`--eval-mc > 0`, and `decision_f_value` is the online low-MC estimate. The
`queries` and `time_sec` fields correspond to online decision making only.

For long HPC jobs, prefer the helper scripts' `ROUTES` and `STATE_DIR`
environment variables. Example:

```bash
ROUTES="gong pham" sbatch runICSlashdot.sh 0.2
```

This skips a completed `OPS`/`STAMP` run and resumes `GongD1` and `Pham22`
from their latest state files when present.

The CSV also keeps support size and memory diagnostics:

- `support_size`
- `graph_mem_mb`
- `algo_mem_mb`

CSV rows include `profit_model`, `profit_mu`, `suit_seed`, `eta_min`,
`eta_max`, and `xi_max`; duplicate-row detection includes these fields.

Run-level stdout also reports:

- `graph_mem_mb`: RSS baseline after the input graph has been loaded
- `algo_mem_mb`: extra peak RSS above that baseline during the algorithm run

This is the number that should be compared across `ops`, `gong`, and `pham`
when memory is reported as a diagnostic; otherwise the common graph storage
would dominate the
measurement and blur the algorithmic memory difference.

## Migration Notes

The code has now been refactored into a prefix-stream benchmark whose active
paper suite is STAMP versus the two non-monotone one-pass streaming baselines.

Recommended direction:

1. keep tightening the prefix-aware objective layer
2. keep scripts and README aligned with the KIC-main plus LT-robustness
   STAMP/Gong/Pham benchmark
3. keep `revenue` and `sensor` out of the active experiment tree unless the
   paper scope changes again

Practical warning:

- if a run prints `B = 0` for a nonzero `B_factor`, the most likely cause is
  that an old binary or old command line is being used
- the current `main.cpp` prints both `B_factor` and the derived `B`
- the batch scripts in this folder should pass `B_factor` directly

## Practical Reminder For Future Codex

Before changing code in this folder, check whether the requested edit is about:

- the experiment harness
- the objective oracle
- preprocessing / data format

If the change is about the paper target, prefer the prefix-stream view.
