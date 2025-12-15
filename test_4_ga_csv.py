#!/usr/bin/env python3
"""
GA-based parallel branch testing with time-based continuous evolution.
Multi-seed version.

Final output directory structure:

benchmark_log_test4_ga_test/
    seed_0/
        file1.csv
        file2.csv
        ...
    seed_1/
    seed_2/
    ...
"""

import os
import csv
import json
import time
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path

from module.sbst_core import instrument_and_load, FitnessCalculator
from BASE.ga import ga


def _ga_worker(args):
    (
        file_path,
        func_name,
        lineno,
        branch_info,
        time_limit,
        random_seed,
        success_threshold,
        pop_size,
        tournament_k,
        elite_ratio,
        gene_mut_p,
        mutation_step_choices,
        ensure_mutation,
        target_outcome,
        use_biased_init,
    ) = args

    worker_verbose = os.environ.get("WORKER_VERBOSE", "0") == "1"
    worker_pid = os.getpid()

    result = {
        "function": func_name,
        "lineno": lineno,
        "outcome": target_outcome,
        "convergence_speed": 0,
        "nfe": 0,
        "best_fitness": float("inf"),
        "best_solution": None,
        "success": False,
        "num_trials": 0,
        "total_time": 0.0,
        "time_to_solution": None,
        "error": None,
        "worker_pid": worker_pid,
    }

    import sys

    old_stdout = sys.stdout
    if not worker_verbose:
        sys.stdout = open(os.devnull, "w")

    try:
        branch_start = time.time()
        random.seed(random_seed)

        source = open(file_path).read()
        namespace, traveler, record, _ = instrument_and_load(source)
        fitness_calc = FitnessCalculator(traveler, record, namespace)
        fitness_calc.evals = 0

        func_obj = namespace[func_name]
        parent_map = traveler.parent_map
        func_info = [f for f in traveler.functions if f.name == func_name][0]

        target_branch_node = branch_info.node
        subject_node = branch_info.subject

        # biased init metadata
        var_constants = getattr(func_info, "var_constants", {}) or {}
        total_constants = list(getattr(func_info, "total_constants", set()) or [])

        rng = random.Random(random_seed)

        # run GA
        ind, fit = ga(
            fitness_calc=fitness_calc,
            func_info=func_info,
            func_obj=func_obj,
            target_branch_node=target_branch_node,
            target_outcome=target_outcome,
            subject_node=subject_node,
            parent_map=parent_map,
            pop_size=pop_size,
            max_gen=10000,
            tournament_k=tournament_k,
            elite_ratio=elite_ratio,
            gene_mut_p=gene_mut_p,
            ensure_mutation=ensure_mutation,
            mutation_step_choices=mutation_step_choices,
            rng=rng,
            use_biased_init=use_biased_init,
            var_constants=var_constants,
            total_constants=total_constants,
            time_limit=time_limit,
            start_time=branch_start,
        )

        nfe = fitness_calc.evals
        generations = ((nfe - 1) // pop_size + 1) if nfe > 0 else 0

        result["convergence_speed"] = generations
        result["nfe"] = nfe
        result["best_fitness"] = fit if fit is not None else float("inf")
        result["best_solution"] = ind
        result["num_trials"] = nfe

        if fit is not None and fit <= success_threshold:
            result["success"] = True
            result["time_to_solution"] = time.time() - branch_start

        result["total_time"] = time.time() - branch_start

    except Exception as e:
        result["error"] = str(e)

    finally:
        if not worker_verbose:
            sys.stdout.close()
            sys.stdout = old_stdout

    return result

def run_parallel_test_with_csv(
    file_path: str,
    output_csv: str,
    time_limit_per_branch: float = 20.0,
    random_seed: int = 42,
    success_threshold: float = 0.0,
    pop_size: int = 10000,
    tournament_k: int = 3,
    elite_ratio: float = 0.1,
    gene_mut_p=None,
    mutation_step_choices=(-3, -2, -1, 1, 2, 3),
    ensure_mutation=True,
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

        for lineno, branch_info in branches.items():
            is_for_loop = lineno in getattr(traveler.tx, "loop_minlen", {})
            is_while_true = lineno in getattr(traveler.tx, "while_always_true", {})

            for target_outcome in [True, False]:
                if skip_for_false:
                    if is_while_true and not target_outcome:
                        continue
                    if is_for_loop and not target_outcome:
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
                        pop_size,
                        tournament_k,
                        elite_ratio,
                        gene_mut_p,
                        mutation_step_choices,
                        ensure_mutation,
                        target_outcome,
                        use_biased_init,
                    )
                )

    if num_workers is None:
        num_workers = cpu_count()

    pool = Pool(processes=num_workers)
    try:
        results = pool.map(_ga_worker, tasks)
        pool.close()
        pool.join()
    finally:
        try:
            pool.close()
            pool.join()
        except:
            pass

    outdir = os.path.dirname(output_csv)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as cf:
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
                "generations",
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
                    "generations": r["convergence_speed"],
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
    output_dir="benchmark_log_test4_ga_test",
    time_limit_per_branch=20.0,
    seeds=[42],
    success_threshold=0.0,
    pop_size=10000,
    num_workers=None,
    skip_for_false=True,
    use_biased_init=True,
):
    source_path = Path(source_dir)

    py_files = [f for f in source_path.rglob("*.py") if "__pycache__" not in str(f)]

    overall_start = time.time()

    for seed in seeds:
        print(f"\n===== Running seed {seed} =====")

        seed_dir = Path(output_dir) / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        for py_file in py_files:
            rel_path = py_file.relative_to(source_path)
            csv_out = seed_dir / rel_path.with_suffix(".csv")
            csv_out.parent.mkdir(parents=True, exist_ok=True)

            print(f"→ Testing {py_file} -> {csv_out}")

            run_parallel_test_with_csv(
                file_path=str(py_file),
                output_csv=str(csv_out),
                time_limit_per_branch=time_limit_per_branch,
                random_seed=seed,
                success_threshold=success_threshold,
                pop_size=pop_size,
                num_workers=num_workers,
                skip_for_false=skip_for_false,
                use_biased_init=use_biased_init,
            )

    total_time = time.time() - overall_start

    cfg = {
        "algorithm": "Genetic Algorithm (multi-seed, continuous evolution)",
        "source_directory": source_dir,
        "output_directory": output_dir,
        "time_limit_per_branch": time_limit_per_branch,
        "seeds": seeds,
        "population_size": pop_size,
        "total_execution_time": total_time,
    }
    with open(Path(output_dir) / "test_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

if __name__ == "__main__":
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, default="./benchmark")
    parser.add_argument(
        "--output", "-o", type=str, default="benchmark_log_test4_ga_test"
    )
    parser.add_argument("--time-limit", "-t", type=float, default=20.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--random-init", action="store_true")
    args = parser.parse_args()

    if "fork" in multiprocessing.get_all_start_methods():
        multiprocessing.set_start_method("fork", force=True)
    else:
        multiprocessing.set_start_method("spawn", force=True)

    print("GA multi-seed config:")
    print(f"source = {args.source}")
    print(f"output = {args.output}")
    print(f"seeds  = {args.seeds}")
    print()

    run_directory_test(
        source_dir=args.source,
        output_dir=args.output,
        time_limit_per_branch=args.time_limit,
        seeds=args.seeds,
        use_biased_init=not args.random_init,
    )
