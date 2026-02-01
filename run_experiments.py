#!/usr/bin/env python3
"""
Main script to run PCP experiments.

Usage:
    # Run on a single instance
    uv run python run_experiments.py --instance instances/ring/ring_n10p0.1s1.txt

    # Run on a family of instances (on easier instances - from the start)
    uv run python run_experiments.py --family ring --max-instances 5

    # Run on harder instances (from the end)
    uv run python run_experiments.py --family ring --max-instances 5 --from-end

    # Run on all families
    uv run python run_experiments.py --all

    # Run with custom time limit in seconds
    uv run python run_experiments.py --family random --time-limit 60

    # NOTE: ILP solver is used by default. Metaheuristic solvers will be added later.
    # ILP uses SCIP backend by default, other backends can be selected with --ilp-backend option.
"""

import argparse
from pathlib import Path

from src.pcp import ILPSolver, PCPInstance
from src.pcp.runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run PCP solver on benchmark instances")

    # Instance selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance", type=Path, help="Path to a single instance file")
    group.add_argument("--family", type=str, help="Instance family to run (e.g., 'ring', 'random')")
    group.add_argument("--all", action="store_true", help="Run on all instance families")

    # Solver parameters, will be extended with metaheuristics later
    parser.add_argument(
        "--solver",
        type=str,
        default="ILP",
        choices=["ILP"],
        help="Solver type to use (default: ILP)",
    )
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
        help="Time limit per instance in seconds (default: 300)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed solver output")

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
    if args.solver == "ILP":
        solver = ILPSolver(
            time_limit_seconds=args.time_limit,
            solver_name=args.ilp_backend,
            verbose=args.verbose,
        )
    elif args.solver == "metaheuristics":
        # TODO: Implement metaheuristic solvers
        raise NotImplementedError("Metaheuristic solvers not yet implemented")
    else:
        raise ValueError(f"Unknown solver type: {args.solver}")

    # Create runner
    runner = ExperimentRunner(solver, output_dir=args.output_dir)

    # Run experiments
    if args.instance:
        # Single instance mode
        print(f"Running on instance: {args.instance}")
        instance = PCPInstance.from_file(args.instance)
        print(f"  {instance}")

        result = solver.solve(instance)
        runner.results.append(result)

        print(f"\nResult: {result.status.value}")
        if result.num_colors is not None:
            print(f"  Partition chromatic number: {result.num_colors}")
            print(f"  Selected vertices: {result.selected_vertices}")
            print(f"  Color assignment: {result.vertex_colors}")

            # Verify solution
            if solver.verify_solution(instance, result):
                print("  Solution verified: VALID")
            else:
                print("  Solution verified: INVALID!")

        print(f"  Runtime: {result.runtime_seconds:.3f}s")

    elif args.family:
        # Single family mode
        runner.run_directory(
            args.instances_dir / args.family,
            max_instances=args.max_instances,
            from_end=args.from_end,
        )
        runner.print_table()
        runner.print_summary()

    else:
        # All families mode
        families = ["ring", "nsfnet", "random", "big-random"]
        runner.run_all_families(
            args.instances_dir,
            families=families,
            max_per_family=args.max_instances,
            from_end=args.from_end,
        )
        runner.print_table()
        runner.print_summary()

    # Save results
    if runner.results:
        runner.save_results_csv(args.output_file, solver_name=args.solver)


if __name__ == "__main__":
    main()
