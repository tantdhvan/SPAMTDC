# ksubconsitent code

This directory contains the experimental harness for the paper draft
`ksubconsitent`.

Current paper scope:

- `STAMP`: the one-consistent Stable Topic Assignment via Marginal Potentials
  algorithm from the active draft. For compatibility with existing HPC runs,
  the executable route is still `ops` and current CSV rows still use route
  label `OPS`.
- `GongD1`: Gong--Liu--Fang-style one-pass non-monotone `k`-submodular
  streaming, specialized to one total capacity dimension and unit costs
- `Pham22`: Pham et al.-style one-pass budgeted non-monotone
  `k`-submodular streaming under the same total capacity
- main diffusion model: KIC with theorem-aligned suitability-adjusted profit
- optional robustness diffusion model: LT for Email/Facebook only, using the
  same suitability-adjusted profit

The active model is prefix-stream:

- at time `t`, only `V_t` is visible
- the objective is `f_t`, not a global full-data objective
- evaluation and marginals must ignore future items outside `V_t`
- the reported profit is `expected topic-union spread + mu * suitability_sum`,
  not the old unit-cost objective `spread - support_size`

## Build

The repository ships separate objective binaries:

```bash
make kic
make lt
make preprockic
```

`kic` and `lt` require OpenMP.

If `make` is not available, the equivalent direct compile command is:

```bash
g++ -std=c++17 -O2 -Wall -Isrc src/main.cpp -o kic -DKFUNC_KIC -fopenmp -lpthread
g++ -std=c++17 -O2 -Wall -Isrc src/main.cpp -o lt -DKFUNC_LT -fopenmp -lpthread
g++ -std=c++17 -O2 -Wall -Isrc src/data/preprocess_kic.cpp -o preprockic -lpthread
```

## Preprocessing

KIC and LT use the shared `.bin` layout:

- `n, m, K, undirected`
- node weights and alpha vectors
- `part_id[n]` as compatibility metadata
- edge list with `K` weights per edge

The important convention is:

- node ids are assigned by first appearance in `edges.txt`
- therefore `node_id == arrival order` in the `.bin`
- `part_id[i]` is set to `i` and is not used by the active paper code

Preprocessor commands:

```bash
./preprockic edges.txt output.bin K [input_undirected=1] [randomize_node=0] [seed=42] [--topic-edge-jitter low high] [--edge-seed seed]
```

If an input edge has one weight or no weight, `--topic-edge-jitter low high`
can generate topic-specific edge probabilities by multiplying the base weight
by a deterministic topic-specific factor in `[low, high]` before normalization.

For the active rerun, regenerate fresh suitability-era graph binaries before
submitting experiment jobs. The old `email.bin` and `fb.bin` remain useful for
format checks, but the paper rerun should use the `*_suit.bin` files so the
dataset provenance matches the topic-specific probability protocol.

```bash
./preprockic data/email.txt data/email_suit.bin 3 0 0 42 --topic-edge-jitter 0.8 1.2 --edge-seed 20260528
./preprockic data/facebook.txt data/fb_suit.bin 3 1 0 42 --topic-edge-jitter 0.8 1.2 --edge-seed 20260528
./preprockic data/Slashdot0902.txt data/Slashdot_suit.bin 3 0 0 42 --topic-edge-jitter 0.8 1.2 --edge-seed 20260528
./preprockic data/youtube.txt data/gplus_suit.bin 3 1 0 42 --topic-edge-jitter 0.8 1.2 --edge-seed 20260528
```

Use `K=3` to match the existing local Email/Facebook binaries. The directedness
flags above follow the usual dataset conventions: Email and Slashdot directed,
Facebook and YouTube/Gplus undirected. The final line uses the local
`youtube.txt` file but keeps the legacy script/output label `gplus`; verify the
raw source name before reporting that dataset in the paper.

## Run

The driver accepts the paper routes directly:

```text
graph.bin route B_factor [options] [csv_path]
```

Examples:

```bash
./kic graph.bin ops 0.20 --alpha 1.505 --profit-mu 1 --suit-seed 20260528 --eta-min 0.2 --eta-max 0.8 --xi-max 0.2 --threads 32 --kic-mc 200 --eval-mc 10000 --seed 42 --csv out.csv
./kic graph.bin gong 0.20 --alpha 0.4 --epsilon 0.1 --profit-mu 1 --suit-seed 20260528 --eta-min 0.2 --eta-max 0.8 --xi-max 0.2 --threads 32 --kic-mc 200 --eval-mc 10000 --seed 42 --csv out.csv
./kic graph.bin pham 0.20 --alpha 0.4 --epsilon 0.1 --profit-mu 1 --suit-seed 20260528 --eta-min 0.2 --eta-max 0.8 --xi-max 0.2 --threads 32 --kic-mc 200 --eval-mc 10000 --seed 42 --csv out.csv
./lt graph.bin ops 0.20 --alpha 1.505 --profit-mu 1 --suit-seed 20260528 --eta-min 0.2 --eta-max 0.8 --xi-max 0.2 --threads 32 --lt-mc 200 --eval-mc 10000 --seed 42 --csv lt_out.csv
```

Each binary has one compiled objective. The optional matching objective token
(`kic` for `./kic`, `lt` for `./lt`) is accepted for old command lines.
`revenue` and `sensor` are intentionally disabled. Valid active routes are
`ops`, `gong`, and `pham`. All active routes emit prefix-wise statistics indexed
by `t` for one fixed `B` per run.

Options:

- `--alpha <double>`: STAMP parameter `d` or the Gong/Pham acceptance parameter
- `--epsilon <double>`: geometric-guess epsilon for `gong` and `pham`
- `--threads <int>`: OpenMP thread count for Monte Carlo objective calls
- `--seed <uint64>`: seed recorded in the output and used for Monte Carlo
  sampling
- `--profit-mu <double>`: suitability scaling parameter; must satisfy
  `profit_mu * eta_max <= 1` for the theorem-aligned nonnegativity condition
- `--suit-seed <uint64>`: seed for deterministic suitability generation
- `--eta-min <double>`, `--eta-max <double>`: bad-topic penalty range
- `--xi-max <double>`: nonnegative good-topic bonus noise cap
- `--mc <size_t>`: decision-oracle Monte Carlo sample count
- `--kic-mc <size_t>`: KIC-compatible alias for `--mc`
- `--lt-mc <size_t>`: LT-compatible alias for `--mc`
- `--eval-mc <size_t>`: high-confidence checkpoint evaluation sample count
  for the compiled objective; set `--eval-mc 0` to keep `f_value` equal to the decision-oracle
  estimate
- `--csv <path>`: append prefix-wise CSV rows to a file
- `--state <path>`: save/load a resumable checkpoint state for `gong` and
  `pham`; ignored by `ops`
- `B_factor` is converted internally as `B = floor(B_factor * |V_n|)`
- CSV route labels are `OPS`, `GongD1`, and `Pham22` for the active suite;
  interpret `OPS` as the legacy label for paper algorithm `STAMP`

The CSV trace is one row per reported checkpoint `t`. Rows are now appended and
flushed as soon as each checkpoint is reached, rather than only after the full
route finishes. For Monte Carlo objectives, the decision oracle uses `--mc`
or the objective-specific alias, while the reported `f_value` is re-evaluated
on the checkpoint assignment with `--eval-mc`; the low-MC online estimate is
preserved as `decision_f_value`.
Query and running-time columns measure the online decision phase, not the
high-MC reporting pass. Duplicate rows with the same graph, route, budget,
parameters, profit parameters, seed, and checkpoint are skipped when a run is
resumed. The four main comparison axes for the paper are `f_value`, `consistency_violations`,
`queries`, and `time_sec`; support and memory are diagnostics. Memory is also
reported on
stdout as a run-level summary with two separate numbers:

- `graph_mem_mb`: peak RSS already reached immediately after loading the graph
- `algo_mem_mb`: additional peak RSS reached during the algorithm, measured
  relative to the post-load graph baseline

Use `algo_mem_mb` for cross-algorithm comparison; it excludes the memory of the
loaded input graph.

For large KIC/LT runs, the parallelism is inside the value oracle: each
`kfunc_evaluate` call parallelizes the Monte Carlo simulations with OpenMP.
Build the target binary with `-fopenmp`, set `OMP_NUM_THREADS` or pass
`--threads`, and request one Slurm task with many CPUs per task rather than
many MPI tasks. The helper scripts already export `OMP_NUM_THREADS`,
`OMP_PROC_BIND`, and `OMP_PLACES`.

The helper shell scripts now take one `B_FACTOR` per invocation, default to
fresh `*_suit_active.csv` outputs, pass the suitability parameters above, and
accept a space-separated `ROUTES` override. For example, to run only baselines
on Slashdot:

```bash
ROUTES="gong pham" sbatch runICSlashdot.sh 0.2
```

The scripts default to the regenerated graph paths:

- `runICEmail.sh`: `data/email_suit.bin`
- `runICFacebook.sh`: `data/fb_suit.bin`
- `runICSlashdot.sh`: `data/Slashdot_suit.bin`
- `runICGplus.sh`: `data/gplus_suit.bin`
- `runLTEmail.sh`: `data/email_suit.bin`
- `runLTFacebook.sh`: `data/fb_suit.bin`

Override `GRAPH=...` only when deliberately using a different binary.

The scripts pass `--state state/<csv-base>.<route>.state` to `gong` and `pham`,
so rerunning the same command continues from the latest saved checkpoint when
the state file is present. `STATE_DIR` can be overridden if the state files
should live elsewhere.

For a normal full run:

```bash
./runICFacebook.sh 0.2
```

For the planned LT robustness runs on the two small datasets:

```bash
make lt
sbatch runLTEmail.sh 0.2
sbatch runLTFacebook.sh 0.2
```

For the active four-axis comparison chart, use:

```bash
python chart/buildActiveMetrics.py out.csv --output out.metrics.pdf
```

## Core files

- `src/main.cpp`: prefix-stream driver
- `src/kfunctions.h`: oracle interface
- `src/prefix_stream.h`: prefix state helper
- `src/algs/potentialswap.h`: STAMP implementation, exposed as route `ops`
- `src/algs/streaming_literature.h`: GongD1 and Pham22 streaming baselines
- `src/algs/result.h`: output struct
- `src/objectvalue/kic.h`
- `src/objectvalue/lt.h`: optional LT robustness oracle
- `src/objectvalue/suitability.h`: deterministic suitability generation and
  suitability-term lookup

Legacy revenue, sensor, fairness, matroid, threshold, quickswap, and recompute
source files have been removed from this tree.
