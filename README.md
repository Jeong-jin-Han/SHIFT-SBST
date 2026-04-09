<h1 align="center">SHIFT-SBST</h1>

<p align="center">
  <b>Sigmoid-Based Heuristic Invertible Fitness-Landscape Transformation for Accelerating SBST</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Institution-KAIST-orange" alt="KAIST"/>
  <img src="https://img.shields.io/badge/Topic-SBST-green" alt="SBST"/>
</p>

---

## Overview

Search-Based Software Testing (SBST) automates test input generation by framing the problem as an optimization task. However, standard algorithms such as Hill Climbing (HC) frequently stall on challenging fitness landscapes — extended plateaus, rugged multimodal terrains, and needle-in-a-haystack configurations — where gradient information is sparse or misleading.

**SHIFT** addresses this by *reshaping* the fitness landscape rather than modifying the search algorithm itself. It detects flat or weakly varying regions, applies an **invertible sigmoid-based compression** to contract them, and enables standard HC to escape stagnation with far fewer evaluations.

---

## Key Contributions

<table>
  <tr>
    <td><b>Invertible Sigmoid Warping</b></td>
    <td>Compresses detected fitness basins into compact intervals in a transformed Z-space, with a closed-form inverse for mapping back to the original domain.</td>
  </tr>
  <tr>
    <td><b>Bidirectional Basin Detection</b></td>
    <td>Identifies the extent of 1D flat regions along each active dimension before applying compression.</td>
  </tr>
  <tr>
    <td><b>Active Dimension Pruning</b></td>
    <td>Tracks stagnation counters per dimension and deactivates irrelevant axes, reducing per-iteration cost from O(n) to O(k).</td>
  </tr>
  <tr>
    <td><b>Informed Restarts</b></td>
    <td>Places restart candidates at detected basin boundaries rather than sampling randomly, converting naive restarts into geometry-aware exploration.</td>
  </tr>
  <tr>
    <td><b>Accumulated Metadata</b></td>
    <td>Preserves compression information across trials, allowing the algorithm to operate on an increasingly informed landscape.</td>
  </tr>
</table>

---

## Repository Structure

```
SHIFT-SBST/
├── module/
│   └── sbst_core.py                    # AST instrumentation, branch-distance computation, fitness function
├── BASE/
│   └── ga.py                           # Genetic Algorithm baseline
├── benchmark/
│   ├── arbitrary1~10.py                # General control-flow programs
│   ├── ex1~7.py                        # Assignment benchmark programs
│   ├── plateau1~3.py                   # Plateau landscape programs
│   ├── needle1~2.py                    # Needle-in-a-haystack programs
│   ├── rugged1~2.py                    # Rugged landscape programs
│   ├── combined1~2.py                  # Mixed-difficulty programs
│   └── test_3/fitness.py              # Synthetic 1D/2D fitness landscapes
├── hill_climb_multiD.py                # Baseline Hill Climbing (HC)
├── compression_hc.py                   # HC-SHIFT: sigmoid warping, basin detection, compression manager
├── test_benchmark_parallel_csv.py      # Evaluate HC-SHIFT on benchmark programs (parallel)
├── test_benchmark_base_parallel_csv.py # Evaluate baseline HC on benchmark programs (parallel)
├── test_benchmark_ga_csv.py            # Evaluate GA on benchmark programs (parallel)
├── test_3_hcc_csv.py                   # Evaluate HC-SHIFT on synthetic landscapes
├── test_3_hc_csv.py                    # Evaluate baseline HC on synthetic landscapes
├── test_3_ga_csv.py                    # Evaluate GA on synthetic landscapes
├── test_3_plot.py                      # Visualize search trajectories on fitness landscapes
├── test_1_summary_csv.py               # Summarize CSV experiment results
├── coverage_generator.py               # Measure branch coverage from generated test inputs
├── run_all.sh                          # Run all experiments (fixed seed)
├── run_all_seed.sh                     # Run all experiments across multiple seeds (41-45)
└── summarize_all.sh                    # Aggregate and summarize all results
```

---

## Requirements

```bash
pip install numpy scipy matplotlib
```

- Python 3.8+
- `numpy`, `scipy`, `matplotlib`
- Standard library: `ast`, `multiprocessing`, `csv`, `json`

---

## Usage

### Run on Benchmark Programs

```bash
# HC-SHIFT (proposed)
python test_benchmark_parallel_csv.py \
    --benchmark-dir benchmark \
    --time-limit 20 \
    --output-dir benchmark_log_hcshift \
    --num-workers 4 \
    --seed 42 \
    --biased-init

# Baseline HC
python test_benchmark_base_parallel_csv.py \
    --benchmark-dir benchmark \
    --time-limit 20 \
    --output-dir benchmark_log_hc

# Baseline GA
python test_benchmark_ga_csv.py \
    --benchmark-dir benchmark \
    --time-limit 20 \
    --output-dir benchmark_log_ga
```

### Run on Synthetic Fitness Landscapes

```bash
# HC-SHIFT on synthetic landscapes
python test_3_hcc_csv.py \
    --fitness-types needle,rugged,plateau,combined \
    --dimensions 1,2 \
    --time-limit 20 \
    --output-dir benchmark_log_test3_hcshift \
    --biased-init

# Baseline HC / GA (same arguments)
python test_3_hc_csv.py ...
python test_3_ga_csv.py ...

# Visualize trajectories
python test_3_plot.py --input-csv <results.csv> --output-dir plots/
```

### Run All Experiments

```bash
./run_all.sh          # Fixed seed
./run_all_seed.sh     # Seeds 41-45 for statistical robustness
./summarize_all.sh    # Aggregate results
```

### Analyze Coverage

```bash
python coverage_generator.py benchmark_log_hcshift
python test_1_summary_csv.py benchmark_log_hcshift
```

---

## Benchmark Categories

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th># Programs</th>
      <th># Branches</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Plateau</td>
      <td>4</td>
      <td>40</td>
      <td>Large flat regions with no gradient information</td>
    </tr>
    <tr>
      <td>Rugged</td>
      <td>3</td>
      <td>22</td>
      <td>Many local optima, chaotic and unpredictable landscape</td>
    </tr>
    <tr>
      <td>Needle-in-a-Haystack</td>
      <td>3</td>
      <td>13</td>
      <td>Near-uniform landscape with a single narrow basin of attraction</td>
    </tr>
    <tr>
      <td>Mixed and Complex</td>
      <td>3</td>
      <td>42</td>
      <td>Combinations of plateau, rugged, and needle challenges</td>
    </tr>
    <tr>
      <td>Other</td>
      <td>25</td>
      <td>226</td>
      <td>General control flow programs</td>
    </tr>
  </tbody>
</table>

---

## Results

Evaluated on **38 benchmark programs** (343 branches) under a **20-second per-branch time budget**:

<table>
  <thead>
    <tr>
      <th>Algorithm</th>
      <th>Initialization</th>
      <th>Coverage</th>
      <th>Runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>HC-SHIFT</b></td>
      <td>biased</td>
      <td><b>95.84%</b> (3642 / 3800)</td>
      <td>3m 39s</td>
    </tr>
    <tr>
      <td>HC-SHIFT</td>
      <td>random</td>
      <td>93.92% (3569 / 3800)</td>
      <td>4m 13s</td>
    </tr>
    <tr>
      <td>HC</td>
      <td>biased</td>
      <td>94.42% (3586 / 3800)</td>
      <td>4m 32s</td>
    </tr>
    <tr>
      <td>HC</td>
      <td>random</td>
      <td>92.60% (3519 / 3800)</td>
      <td>5m 03s</td>
    </tr>
    <tr>
      <td>GA</td>
      <td>biased</td>
      <td>92.60% (3519 / 3800)</td>
      <td>4m 55s</td>
    </tr>
    <tr>
      <td>GA</td>
      <td>random</td>
      <td>91.87% (3491 / 3800)</td>
      <td>4m 54s</td>
    </tr>
  </tbody>
</table>

On structurally difficult benchmarks, HC-SHIFT converges in **2–8 trials on average**, while HC and GA require **hundreds of thousands of trials** and still fail to achieve full coverage.
