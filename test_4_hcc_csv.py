#!/usr/bin/env python3
"""
Hill-Climbing with Sigmoid Compression
Multi-seed Support + directory structure:

benchmark_log_test4_hcc_test/
    seed_0/
        <mirrored benchmark files>.csv
    seed_1/
    ...
"""

import os
import sys
import time
import random
import csv
import json
import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count
from module.sbst_core import instrument_and_load, FitnessCalculator
from compression_hc import hill_climb_with_compression_nd_code, CompressionManagerND


def test_single_branch_with_metrics(args):
    (
        file_path,
        func_name,
        lineno,
        branch_data,
        time_limit,
        random_seed,
        success_threshold,
        initial_low,
        initial_high,
        max_iterations,
        basin_max_search,
        target_outcome,
        use_biased_init,
    ) = args

    worker_pid = os.getpid()
    outcome_str = "True" if target_outcome else "False"

    worker_verbose = os.environ.get("WORKER_VERBOSE", "0") == "1"

    branch_start_time = time.time()

    random.seed(random_seed)

    source = open(file_path).read()
    namespace, traveler, record, _ = instrument_and_load(source)

    fitness_calc = FitnessCalculator(traveler, record, namespace)
    fitness_calc.evals = 0

    parent_map = traveler.parent_map
    func_obj = namespace[func_name]
    branch_info = traveler.branches[func_name][lineno]
    target_branch_node = branch_info.node
    subject_node = branch_info.subject
    func_info = [f for f in traveler.functions if f.name == func_name][0]
    dim = len(func_info.args)
    func_args = func_info.args

    var_constants = getattr(func_info, "var_constants", {}) or {}
    total_constants = list(getattr(func_info, "total_constants", set()) or {})

    def sample_initial_arg(arg_name, low, high):
        if not use_biased_init:
            return random.randint(low, high)

        if not total_constants and not var_constants:
            return random.randint(low, high)

        if random.random() < 0.2:
            return random.randint(low, high)

        consts = list(var_constants.get(arg_name, []))
        if not consts:
            consts = total_constants
        if not consts:
            return random.randint(low, high)

        center = random.choice(consts)
        sigma = max(1, int(0.01 * (high - low)))
        val = int(random.gauss(center, sigma))
        return min(max(val, low), high)

    branch_cm = CompressionManagerND(dim, steepness=5.0)

    total_steps = 0
    best_fitness = float("inf")
    best_solution = None
    branch_success = False
    trial_results = []
    time_to_solution = None
    trial = 0

    while True:
        elapsed = time.time() - branch_start_time
        if elapsed >= time_limit:
            break
        if branch_success:
            break

        initial = [
            sample_initial_arg(arg_name, initial_low, initial_high)
            for arg_name in func_args
        ]

        init_f = fitness_calc.fitness_for_candidate(
            func_obj,
            initial,
            target_branch_node,
            target_outcome,
            subject_node,
            parent_map,
        )

        old_stdout = sys.stdout
        if not worker_verbose:
            sys.stdout = open(os.devnull, "w")

        try:
            traj, branch_cm = hill_climb_with_compression_nd_code(
                fitness_calc,
                func_obj,
                target_branch_node,
                target_outcome,
                subject_node,
                parent_map,
                initial,
                dim,
                max_iterations=max_iterations,
                basin_max_search=basin_max_search,
                global_min_threshold=1e-6,
                cm=branch_cm,
                time_limit=time_limit,
                start_time=branch_start_time,
            )
        finally:
            if not worker_verbose:
                sys.stdout.close()
                sys.stdout = old_stdout

        final_point, final_f, used_comp = traj[-1]
        steps = len(traj)

        total_steps += steps
        if final_f < best_fitness:
            best_fitness = final_f
            best_solution = list(final_point)

        trial_results.append(
            {
                "trial": trial,
                "initial": initial,
                "init_f": init_f,
                "final": list(final_point),
                "final_f": final_f,
                "steps": steps,
            }
        )

        if final_f <= success_threshold:
            branch_success = True
            time_to_solution = time.time() - branch_start_time

        trial += 1

    total_nfe = fitness_calc.evals
    total_time = time.time() - branch_start_time

    return {
        "function": func_name,
        "lineno": lineno,
        "outcome": target_outcome,
        "convergence_speed": total_steps,
        "nfe": total_nfe,
        "best_fitness": best_fitness,
        "best_solution": best_solution,
        "success": branch_success,
        "num_trials_run": len(trial_results),
        "total_time": total_time,
        "time_to_solution": time_to_solution,
        "trial_details": trial_results,
    }


def run_parallel_test_with_csv(
    file_path,
    output_csv,
    time_limit_per_branch=20.0,
    random_seed=42,
    success_threshold=0.0,
    initial_low=-10000,
    initial_high=10000,
    max_iterations=10,
    basin_max_search=1000,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True,
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
            func_low = func_info.min_const
            func_high = func_info.max_const
        else:
            func_low = initial_low
            func_high = initial_high

        for lineno, branch_info in branches.items():
            is_for_loop = lineno in getattr(traveler.tx, "loop_minlen", {})
            is_while_true = lineno in getattr(traveler.tx, "while_always_true", {})

            for target_outcome in [True, False]:
                if skip_for_false:
                    if is_while_true and target_outcome is False:
                        continue
                    if is_for_loop and target_outcome is False:
                        continue

                tasks.append(
                    (
                        file_path,
                        func_name,
                        lineno,
                        branch_info,
                        time_limit_per_branch,
                        random_seed,
                        success_threshold,
                        func_low,
                        func_high,
                        max_iterations,
                        basin_max_search,
                        target_outcome,
                        use_biased_init,
                    )
                )

    if num_workers is None:
        num_workers = cpu_count()

    pool = Pool(processes=num_workers)
    try:
        results = pool.map(test_single_branch_with_metrics, tasks)
        pool.close()
        pool.join()
    finally:
        try:
            pool.close()
            pool.join()
        except:
            pass

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv, "w", newline="") as cf:
        writer = csv.DictWriter(
            cf,
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
                    "num_trials": r["num_trials_run"],
                    "total_time": f"{r['total_time']:.4f}",
                    "time_to_solution": (
                        f"{r['time_to_solution']:.4f}"
                        if r["time_to_solution"]
                        else "N/A"
                    ),
                }
            )

    return results, output_csv


# Multi-seed directory test runner
def run_directory_test(
    source_dir,
    output_dir="benchmark_log_test4_hcc_test",
    time_limit_per_branch=20.0,
    seeds=[42],
    success_threshold=0.0,
    initial_low=-10000,
    initial_high=10000,
    max_iterations=10,
    basin_max_search=1000,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True,
):

    source_path = Path(source_dir)
    all_py = [f for f in source_path.rglob("*.py") if "__pycache__" not in str(f)]

    overall_start = time.time()

    for seed in seeds:
        print(f"\n===== Running seed {seed} =====")

        seed_dir = Path(output_dir) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for py_file in all_py:
            rel = py_file.relative_to(source_path)
            csv_out = seed_dir / rel.with_suffix(".csv")
            csv_out.parent.mkdir(parents=True, exist_ok=True)

            print(f"→ Testing {py_file} -> {csv_out}")

            run_parallel_test_with_csv(
                file_path=str(py_file),
                output_csv=str(csv_out),
                time_limit_per_branch=time_limit_per_branch,
                random_seed=seed,
                success_threshold=success_threshold,
                initial_low=initial_low,
                initial_high=initial_high,
                max_iterations=max_iterations,
                basin_max_search=basin_max_search,
                num_workers=num_workers,
                skip_for_false=skip_for_false,
                use_biased_init=use_biased_init,
            )

    total_time = time.time() - overall_start
    cfg = {
        "algorithm": "HillClimbing+Compression (multi-seed)",
        "source_directory": source_dir,
        "output_directory": output_dir,
        "time_limit_per_branch": time_limit_per_branch,
        "seeds": seeds,
        "total_time": total_time,
    }
    with open(Path(output_dir) / "test_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":

    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, default="./benchmark")
    parser.add_argument(
        "--output", "-o", type=str, default="benchmark_log_test4_hcc_test"
    )
    parser.add_argument("--time-limit", "-t", type=float, default=20.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--random-init", action="store_true")
    args = parser.parse_args()

    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    else:
        multiprocessing.set_start_method("spawn", force=True)

    print("HC multi-seed config:")
    print(f"source = {args.source}")
    print(f"output = {args.output}")
    print(f"seeds = {args.seeds}")
    print()

    run_directory_test(
        source_dir=args.source,
        output_dir=args.output,
        time_limit_per_branch=args.time_limit,
        seeds=args.seeds,
        use_biased_init=not args.random_init,
    )
