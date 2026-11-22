"""Simulation comparison of Section 5.2: DualACO vs Single-ACO, MILP and PSO.

    python compare.py [--tasks 5 10 20] [--runs 5] [--seed 2026]

Each run samples a new task instance (same procedure as main.py), computes the
A* cost matrices once and evaluates every method on them. Stochastic methods
use the instance seed as run seed. Results go to comparison_results.csv.
"""
import argparse
import csv
import time

import numpy as np

import main
import baselines


def sample_instance(m_tasks, seed):
    rng = np.random.RandomState(seed)
    tasks = np.zeros((m_tasks, 2))
    for i in range(m_tasks):
        tasks[i] = main.sample_valid_position(rng, main.occ_grid, main.origin, main.res,
                                              main.task_x_min, main.task_x_max,
                                              main.task_y_min, main.task_y_max)
    drops = np.zeros((m_tasks, 2))
    for i in range(m_tasks):
        tx, ty = tasks[i]
        drops[i] = main.sample_valid_position(rng, main.occ_grid, main.origin, main.res,
                                              tx - main.offset_range, tx + main.offset_range,
                                              ty - main.offset_range, ty + main.offset_range)
    return tasks, drops

def use_instance(tasks, drops, dist):
    # run_aco() reads the instance from main's module globals
    main.m_tasks = len(tasks)
    main.task_positions_world = tasks
    main.dropoff_positions_world = drops
    (main.dist_robot_to_task, main.dist_task_to_drop,
     main.dist_drop_to_task, main.dist_task_to_task) = dist

def run_dualaco(seed):
    _, total, routes, _, _ = main.run_aco(seed)
    return routes, total

def run_methods(seed, dist):
    r2t, t2d, d2t, _ = dist
    methods = {
        "DualACO": lambda: run_dualaco(seed),
        "Single-ACO": lambda: baselines.run_single_aco(seed, r2t, t2d, d2t),
        "PSO": lambda: baselines.run_pso(seed, r2t, t2d, d2t),
        "MILP": lambda: baselines.run_milp(r2t, t2d, d2t),
    }
    results = {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        routes, _ = fn()
        runtime = time.perf_counter() - t0
        # every method is scored with the same A* cost, Eq. (8)
        results[name] = (baselines.route_cost(routes, r2t, t2d, d2t), runtime)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="comparison_results.csv")
    args = parser.parse_args()

    rows = []
    for m_tasks in args.tasks:
        for run in range(args.runs):
            seed = args.seed + 1000 * m_tasks + run
            tasks, drops = sample_instance(m_tasks, seed)
            dist = main.compute_astar_distance_matrices(main.occ_grid, main.origin, main.res,
                                                        main.robot_positions_world, tasks, drops)
            use_instance(tasks, drops, dist)
            for name, (total, runtime) in run_methods(seed, dist).items():
                rows.append({"tasks": m_tasks, "run": run + 1, "instance_seed": seed, "method": name,
                             "total_distance": total, "runtime_s": runtime})
                print(f"[{m_tasks} tasks, run {run + 1}] {name}: {total:.2f} m ({runtime:.2f} s)")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] Saved results to: {args.out}")

    print("\nTotal distance (m), mean ± sample std over runs")
    for m_tasks in args.tasks:
        dist_by_method = {}
        for r in rows:
            if r["tasks"] == m_tasks:
                dist_by_method.setdefault(r["method"], []).append(r["total_distance"])
        for name, vals in dist_by_method.items():
            vals = np.array(vals)
            std = vals.std(ddof=1) if len(vals) > 1 else 0.0
            print(f"  {m_tasks:>2} tasks  {name:<10} {vals.mean():7.2f} ± {std:.2f}")

        # Table 4: paired Wilcoxon signed-rank test and paired Cohen's d, DualACO vs PSO
        diff = np.array(dist_by_method["DualACO"]) - np.array(dist_by_method["PSO"])
        if len(diff) > 1 and np.any(diff != 0):
            d = diff.mean() / diff.std(ddof=1)
            try:
                from scipy.stats import wilcoxon
                p = f"{wilcoxon(diff).pvalue:.4f}"
            except ImportError:
                p = "n/a (scipy not installed)"
            print(f"  {m_tasks:>2} tasks  DualACO vs PSO: Wilcoxon p = {p}, paired Cohen's d = {d:.3f}")
