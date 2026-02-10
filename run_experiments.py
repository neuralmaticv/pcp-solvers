#!/usr/bin/env python3
"""
Main script to run PCP experiments.

Usage:
    # Run on a single instance with ILP
    uv run python run_experiments.py --instance instances/ring/ring_n10p0.1s1.txt

    # Run on a family of instances (on easier instances - from the start)
    uv run python run_experiments.py --family ring --max-instances 5

    # Run on harder instances (from the end)
    uv run python run_experiments.py --family ring --max-instances 5 --from-end

    # Run on all families
    uv run python run_experiments.py --all

    # Run with custom time limit in seconds (ILP)
    uv run python run_experiments.py --family random --time-limit 60

    # Run with ACO solver (default: 20 independent runs per instance)
    uv run python run_experiments.py --solver ACO --family ring

    # Run ACO on first 10 instances only (for quick testing)
    uv run python run_experiments.py --solver ACO --family ring --max-instances 10

    # Run ACO with custom parameters (10 runs per instance, 50 iterations each)
    uv run python run_experiments.py --solver ACO --family ring --aco-runs 10 --aco-iterations 50

    # Run with TabuSearch solver (default: 20 independent runs per instance)
    uv run python run_experiments.py --solver TabuSearch --family ring

    # Run TabuSearch with custom parameters (10 runs, Fend=10)
    uv run python run_experiments.py --solver TabuSearch --family ring --ts-runs 10 --ts-iterations 10

    # ILP uses SCIP backend by default, other backends can be selected with --ilp-backend option.
"""

import argparse
from pathlib import Path
from typing import Union

from src.pcp import ACOResult, ACOSolver, ILPSolver, PCPInstance, SolverResult, TabuSearchResult, TabuSearchSolver
from src.pcp.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run PCP solver on benchmark instances")

    # Instance selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance", type=Path, help="Path to a single instance file")
    group.add_argument("--family", type=str, help="Instance family to run (e.g., 'ring', 'random')")
    group.add_argument("--all", action="store_true", help="Run on all instance families")

    # Solver selection
    parser.add_argument(
        "--solver",
        type=str,
        default="ILP",
        choices=["ILP", "ACO", "TabuSearch"],
        help="Solver type to use (default: ILP)",
    )

    # ILP-specific parameters
    parser.add_argument(
        "--ilp-backend",
        type=str,
        default="SCIP",
        choices=["SCIP", "CBC"],
        help="ILP solver backend when using ILP solver (default: SCIP)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Time limit per instance in seconds for ILP (default: 300)",
    )

    # ACO-specific parameters
    parser.add_argument(
        "--aco-iterations",
        type=int,
        default=100,
        help="Number of ACO iterations per run (default: 100)",
    )
    parser.add_argument(
        "--aco-runs",
        type=int,
        default=20,
        help="Number of independent ACO runs per instance (default: 20)",
    )
    parser.add_argument(
        "--aco-alpha",
        type=float,
        default=1.0,
        help="ACO pheromone importance (default: 1.0)",
    )
    parser.add_argument(
        "--aco-beta",
        type=float,
        default=2.0,
        help="ACO heuristic importance (default: 2.0)",
    )
    parser.add_argument(
        "--aco-rho",
        type=float,
        default=0.1,
        help="ACO evaporation rate (default: 0.1)",
    )
    parser.add_argument(
        "--aco-seed",
        type=int,
        default=None,
        help="ACO base random seed for reproducibility (default: None = random)",
    )

    # TabuSearch-specific parameters
    parser.add_argument(
        "--ts-iterations",
        type=int,
        default=5,
        help="TabuSearch max_iter_factor (Fend in paper, default: 5)",
    )
    parser.add_argument(
        "--ts-runs",
        type=int,
        default=20,
        help="Number of independent TabuSearch runs per instance (default: 20)",
    )
    parser.add_argument(
        "--ts-tabu-min",
        type=float,
        default=0.0,
        help="TabuSearch min tabu tenure as fraction of colors (default: 0.0)",
    )
    parser.add_argument(
        "--ts-tabu-max",
        type=float,
        default=0.5,
        help="TabuSearch max tabu tenure as fraction of colors (default: 0.5)",
    )
    parser.add_argument(
        "--ts-seed",
        type=int,
        default=None,
        help="TabuSearch base random seed for reproducibility (default: None = random)",
    )

    # Common parameters
    parser.add_argument("--verbose", action="store_true", help="Print detailed solver output")

    # Parallel processing
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all CPU cores). Set to 1 for sequential processing.",
    )

    # Experiment parameters
    parser.add_argument("--max-instances", type=int, help="Maximum instances to run (for testing)")
    parser.add_argument(
        "--from-end",
        action="store_true",
        help="Select instances from end of sorted list (harder instances)",
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        default=Path("instances"),
        help="Directory containing instance families",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for output files",
    )
    parser.add_argument("--output-file", type=str, help="Output CSV filename (default: auto-generated)")

    args = parser.parse_args()

    # Create solver based on type
    solver: Union[ILPSolver, ACOSolver, TabuSearchSolver]
    if args.solver == "ILP":
        solver = ILPSolver(
            time_limit_seconds=args.time_limit,
            solver_name=args.ilp_backend,
            verbose=args.verbose,
        )
    elif args.solver == "ACO":
        solver = ACOSolver(
            num_iterations=args.aco_iterations,
            num_runs=args.aco_runs,
            alpha=args.aco_alpha,
            beta=args.aco_beta,
            rho=args.aco_rho,
            base_seed=args.aco_seed,
            verbose=args.verbose,
        )
    elif args.solver == "TabuSearch":
        solver = TabuSearchSolver(
            tabu_tenure_range=(args.ts_tabu_min, args.ts_tabu_max),
            max_iter_factor=args.ts_iterations,
            num_runs=args.ts_runs,
            base_seed=args.ts_seed,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unknown solver type: {args.solver}")

    # Create unified runner
    runner = ExperimentRunner(solver, output_dir=args.output_dir, num_workers=args.workers)

    # Run experiments
    if args.instance:
        # Single instance mode — init output and run
        runner.init_output(args.output_file)

        print(f"Running on instance: {args.instance}")
        instance = PCPInstance.from_file(args.instance)
        print(f"  {instance}")

        result = solver.solve(instance)
        runner.results.append(result)
        runner._append_result_csv(result)

        # Print result based on solver type
        if isinstance(result, SolverResult) and isinstance(solver, ILPSolver):
            print(f"\nResult: {result.status.value}")
            if result.num_colors is not None:
                print(f"  Partition chromatic number: {result.num_colors}")
                print(f"  Selected vertices: {result.selected_vertices}")
                print(f"  Color assignment: {result.vertex_colors}")
                if solver.verify_solution(instance, result):
                    print("  Solution verified: VALID")
                else:
                    print("  Solution verified: INVALID!")
            print(f"  Runtime: {result.runtime_seconds:.3f}s")
        elif isinstance(result, ACOResult) and isinstance(solver, ACOSolver):
            print(f"\nResult ({result.num_runs} runs):")
            print(f"  Best colors: {result.best_colors}")
            print(f"  Avg colors: {result.avg_colors:.2f}")
            print(f"  Std colors: {result.std_colors:.2f}")
            print(f"  All runs: {result.all_colors}")
            if solver.verify_solution(instance, result):
                print("  Best solution verified: VALID")
            else:
                print("  Best solution verified: INVALID!")
            print(f"  Total runtime: {result.total_runtime_seconds:.3f}s")
        elif isinstance(result, TabuSearchResult) and isinstance(solver, TabuSearchSolver):
            print(f"\nResult ({result.num_runs} runs):")
            print(f"  Best colors: {result.best_colors}")
            print(f"  Avg colors: {result.avg_colors:.2f}")
            print(f"  Std colors: {result.std_colors:.2f}")
            print(f"  All runs: {result.all_colors}")
            if solver.verify_solution(instance, result):
                print("  Best solution verified: VALID")
            else:
                print("  Best solution verified: INVALID!")
            print(f"  Total runtime: {result.total_runtime_seconds:.3f}s")

    elif args.family:
        # Single family mode — init output, then run (results appended incrementally)
        runner.init_output(args.output_file)
        runner.run_directory(
            args.instances_dir / args.family,
            max_instances=args.max_instances,
            from_end=args.from_end,
        )
        runner.print_table()
        runner.print_summary()

    else:
        # All families mode — init output, then run (results appended incrementally)
        runner.init_output(args.output_file)
        families = ["ring", "nsfnet", "random", "big-random"]
        runner.run_all_families(
            args.instances_dir,
            families=families,
            max_per_family=args.max_instances,
            from_end=args.from_end,
        )
        runner.print_table()
        runner.print_summary()


if __name__ == "__main__":
    main()
