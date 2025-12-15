#!/usr/bin/env python3
"""
Baseline Hill Climbing (NO COMPRESSION) with multi-seed parallel testing.
Seed structure identical to test_4_hcc_csv.py and test_4_ga_csv.py.
"""

import os
import sys
import time
import random
import csv
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

from module.sbst_core import instrument_and_load, FitnessCalculator
from hill_climb_multiD import hill_climb_simple_nd_code


def test_single_branch_baseline_with_metrics(args):
    (
        file_path,
        func_name,
        lineno,
        branch_data,
        time_limit,
        seed,
        success_threshold,
        initial_low,
        initial_high,
        max_steps,
        target_outcome,
        use_biased_init,
    ) = args

    worker_pid = os.getpid()
    random.seed(seed)
    branch_start_time = time.time()

    # Load instrumented code
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)

    fitness_calc = FitnessCalculator(traveler, record, namespace)
    fitness_calc.evals = 0

    parent_map = traveler.parent_map
    func_obj = namespace[func_name]
    branch_info = traveler.branches[func_name][lineno]
    target_node = branch_info.node
    subject_node = branch_info.subject

    func_info = [f for f in traveler.functions if f.name == func_name][0]
    dim = len(func_info.args)
    func_args = func_info.args

    var_constants = getattr(func_info, "var_constants", {}) or {}
    total_constants = list(getattr(func_info, "total_constants", set()) or [])

    def sample_initial(arg_name, low, high):
        if not use_biased_init:
            return random.randint(low, high)

        if not total_constants and not var_constants:
            return random.randint(low, high)

        if random.random() < 0.2:
            return random.randint(low, high)

        const_list = list(var_constants.get(arg_name, [])) or total_constants
        center = random.choice(const_list)
        span = max(1, high - low)
        sigma = max(1, int(0.01 * span))
        val = int(random.gauss(center, sigma))
        return max(min(val, high), low)

    total_steps = 0
    best_fitness = float("inf")
    best_solution = None
    branch_success = False
    time_to_solution = None
    num_trials = 0

    while True:
        elapsed = time.time() - branch_start_time
        if elapsed >= time_limit:
            break
        if branch_success:
            break

        initial = [sample_initial(a, initial_low, initial_high) for a in func_args]

        init_fit = fitness_calc.fitness_for_candidate(
            func_obj, initial, target_node, target_outcome, subject_node, parent_map
        )

        traj = hill_climb_simple_nd_code(
            fitness_calc,
            func_obj,
            target_node,
            target_outcome,
            subject_node,
            parent_map,
            initial,
            dim,
            max_steps=max_steps,
            time_limit=time_limit,
            start_time=branch_start_time,
        )

        final_point, final_fit = traj[-1]
        steps_this = len(traj)
        total_steps += steps_this

        if final_fit < best_fitness:
            best_fitness = final_fit
            best_solution = list(final_point)

        if final_fit <= success_threshold:
            branch_success = True
            time_to_solution = time.time() - branch_start_time

        num_trials += 1

    total_time = time.time() - branch_start_time

    return {
        "function": func_name,
        "lineno": lineno,
        "outcome": target_outcome,
        "convergence_speed": total_steps,
        "nfe": fitness_calc.evals,
        "best_fitness": best_fitness,
        "best_solution": best_solution,
        "success": branch_success,
        "num_trials": num_trials,
        "total_time": total_time,
        "time_to_solution": time_to_solution,
    }


def run_parallel_baseline_for_file(
    file_path,
    output_csv,
    time_limit_per_branch,
    seed,
    success_threshold,
    initial_low,
    initial_high,
    max_steps,
    num_workers,
    skip_for_false,
    use_biased_init,
):
    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)

    tasks = []
    for func_info in traveler.functions:
        func_name = func_info.name
        branches = traveler.branches.get(func_name, {})
        if not branches:
            continue

        func_range = func_info.max_const - func_info.min_const
        if func_range >= 10:
            ilow = func_info.min_const
            ihigh = func_info.max_const
        else:
            ilow = initial_low
            ihigh = initial_high

        for lineno, branch_info in branches.items():
            is_for_loop = lineno in getattr(traveler.tx, "loop_minlen", {})
            is_while_true = lineno in getattr(traveler.tx, "while_always_true", {})

            for outcome in [True, False]:
                if skip_for_false and is_while_true and not outcome:
                    continue
                if skip_for_false and is_for_loop and not outcome:
                    continue

                tasks.append(
                    (
                        file_path,
                        func_name,
                        lineno,
                        branch_info,
                        time_limit_per_branch,
                        seed,
                        success_threshold,
                        ilow,
                        ihigh,
                        max_steps,
                        outcome,
                        use_biased_init,
                    )
                )

    if num_workers is None:
        num_workers = cpu_count()

    with Pool(processes=num_workers) as pool:
        results = pool.map(test_single_branch_baseline_with_metrics, tasks)

    # Write CSV
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "function",
                "lineno",
                "outcome",
                "convergence_speed",
                "nfe",
                "best_fitness",
                "best_solution",
                "success",
                "num_trials",
                "total_time",
                "time_to_solution",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "function": r["function"],
                    "lineno": r["lineno"],
                    "outcome": r["outcome"],
                    "convergence_speed": r["convergence_speed"],
                    "nfe": r["nfe"],
                    "best_fitness": r["best_fitness"],
                    "best_solution": str(r["best_solution"]),
                    "success": r["success"],
                    "num_trials": r["num_trials"],
                    "total_time": f"{r['total_time']:.3f}",
                    "time_to_solution": (
                        f"{r['time_to_solution']:.3f}"
                        if r["time_to_solution"]
                        else "N/A"
                    ),
                }
            )

    return results


# Multi-seed directory test runner
def run_directory_test_baseline_multiseed(
    source_dir,
    output_dir,
    seeds,
    time_limit_per_branch=20.0,
    success_threshold=0.0,
    initial_low=-1000,
    initial_high=1000,
    max_steps=2000,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True,
):
    source_dir = Path(source_dir)
    py_files = [f for f in source_dir.rglob("*.py") if "__pycache__" not in str(f)]

    for seed in seeds:
        print(f"\n========== SEED {seed} ==========")
        seed_dir = Path(output_dir) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for py_file in py_files:
            rel = py_file.relative_to(source_dir)
            out_csv = seed_dir / rel.with_suffix(".csv")
            out_csv.parent.mkdir(parents=True, exist_ok=True)

            print(f"Running baseline HC on {py_file} → {out_csv}")
            run_parallel_baseline_for_file(
                file_path=str(py_file),
                output_csv=str(out_csv),
                time_limit_per_branch=time_limit_per_branch,
                seed=seed,
                success_threshold=success_threshold,
                initial_low=initial_low,
                initial_high=initial_high,
                max_steps=max_steps,
                num_workers=num_workers,
                skip_for_false=skip_for_false,
                use_biased_init=use_biased_init,
            )

    # Save config
    config_path = Path(output_dir) / "test_config.json"
    config_data = {
        "algorithm": "Baseline Hill Climbing (NO COMPRESSION)",
        "initialization": "biased" if use_biased_init else "random",
        "seeds": seeds,
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "time_limit_per_branch": time_limit_per_branch,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n⭐ All seeds completed. Results saved under: {output_dir}")


if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(
        description="Baseline Hill Climbing (NO COMPRESSION) – Multi-seed"
    )
    parser.add_argument("--source", "-s", type=str, default="./benchmark")
    parser.add_argument(
        "--output", "-o", type=str, default="benchmark_log_test4_hc_test"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--time-limit", "-t", type=float, default=20.0)

    args = parser.parse_args()

    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    else:
        multiprocessing.set_start_method("spawn", force=True)

    use_biased = not args.random_init

    print("\n=== BASELINE HC NO COMPRESSION ===")
    print("Source:", args.source)
    print("Output:", args.output)
    print("Seeds:", args.seeds)
    print("Init:", "RANDOM" if args.random_init else "BIASED")

    run_directory_test_baseline_multiseed(
        source_dir=args.source,
        output_dir=args.output,
        seeds=args.seeds,
        time_limit_per_branch=args.time_limit,
        use_biased_init=use_biased,
    )
