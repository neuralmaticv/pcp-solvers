"""
Experiment runner for PCP instances.

Supports ILP, ACO, and TabuSearch solvers with unified interface.
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Protocol, Union

from .aco_solver import ACOResult, ACOSolver
from .ilp_solver import ILPSolver, SolverResult
from .instance import PCPInstance
from .tabu_search import TabuSearchResult, TabuSearchSolver


# Type alias for results
Result = Union[SolverResult, ACOResult, TabuSearchResult]


class Solver(Protocol):
    """Protocol for PCP solvers."""

    def solve(self, instance: PCPInstance) -> Result: ...
    def verify_solution(self, instance: PCPInstance, result: Result) -> bool: ...


class ExperimentRunner:
    """Runs experiments on PCP instances and collects results.

    Supports ILP, ACO, and TabuSearch solvers with appropriate output formatting.
    """

    def __init__(
        self,
        solver: Union[ILPSolver, ACOSolver, TabuSearchSolver],
        output_dir: Path | None = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            solver: The PCP solver to use (ILPSolver, ACOSolver, or TabuSearchSolver)
            output_dir: Directory for output files (default: current directory)
        """
        self.solver = solver
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.results: list[Result] = []
        if isinstance(solver, ACOSolver):
            self._solver_type = "ACO"
        elif isinstance(solver, TabuSearchSolver):
            self._solver_type = "TabuSearch"
        else:
            self._solver_type = "ILP"

    def run_instance(self, filepath: Path) -> Result:
        """Run solver on a single instance."""
        instance = PCPInstance.from_file(filepath)
        result = self.solver.solve(instance)
        self.results.append(result)
        return result

    def run_directory(
        self,
        directory: Path,
        pattern: str = "*.txt",
        max_instances: int | None = None,
        from_end: bool = False,
    ) -> list[Result]:
        """
        Run solver on all instances in a directory.

        Args:
            directory: Directory containing instance files
            pattern: Glob pattern for instance files
            max_instances: Maximum number of instances to run (for testing)
            from_end: If True, select instances from the end (harder instances first)

        Returns:
            List of results
        """
        directory = Path(directory)
        files = sorted(directory.glob(pattern))

        if max_instances:
            if from_end:
                files = files[-max_instances:]
            else:
                files = files[:max_instances]

        results = []
        for i, filepath in enumerate(files):
            print(f"[{i + 1}/{len(files)}] Processing {filepath.name}...", end=" ")
            sys.stdout.flush()

            try:
                result = self.run_instance(filepath)
                self._print_result_line(result)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            results.append(result)

        return results

    def _print_result_line(self, result: Result) -> None:
        """Print a single result line based on solver type."""
        if isinstance(result, (ACOResult, TabuSearchResult)):
            print(
                f"best={result.best_colors}, avg={result.avg_colors:.2f}, "
                f"std={result.std_colors:.2f} in {result.total_runtime_seconds:.2f}s"
            )
        else:
            print(f"{result.status.value} - {result.num_colors} colors in {result.runtime_seconds:.2f}s")

    def run_all_families(
        self,
        instances_dir: Path,
        families: list[str] | None = None,
        max_per_family: int | None = None,
        from_end: bool = False,
    ) -> dict[str, list[Result]]:
        """
        Run solver on all instance families.

        Args:
            instances_dir: Root directory containing family subdirectories
            families: List of family names to run (default: all)
            max_per_family: Maximum instances per family (for testing)
            from_end: If True, select instances from the end (harder instances first)

        Returns:
            Dictionary mapping family name to results
        """
        instances_dir = Path(instances_dir)

        if families is None:
            families = [d.name for d in instances_dir.iterdir() if d.is_dir()]

        all_results = {}
        for family in families:
            family_dir = instances_dir / family
            if not family_dir.exists():
                print(f"Warning: Family directory {family} not found, skipping.")
                continue

            print(f"\n{'=' * 60}")
            print(f"Running family: {family}")
            print(f"{'=' * 60}")

            results = self.run_directory(family_dir, max_instances=max_per_family, from_end=from_end)
            all_results[family] = results

        return all_results

    def save_results_csv(self, filename: str | None = None, solver_name: str | None = None) -> Path:
        """
        Save all results to a CSV file.

        Args:
            filename: Output filename (default: results_SOLVER_TIMESTAMP.csv)
            solver_name: Name of the solver used (included in default filename)

        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            solver_part = f"{solver_name or self._solver_type}_"
            filename = f"results_{solver_part}{timestamp}.csv"

        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header based on solver type
            if self._solver_type in ["ACO", "TabuSearch"]:
                writer.writerow(
                    ["instance", "vertices", "edges", "partitions", "runs", "best", "avg", "std", "total_time_s"]
                )
            else:
                writer.writerow(
                    ["instance", "vertices", "edges", "partitions", "status", "colors", "runtime_s", "gap"]
                )

            # Write data rows
            for result in self.results:
                if isinstance(result, (ACOResult, TabuSearchResult)):
                    writer.writerow(
                        [
                            result.instance_name,
                            result.num_vertices,
                            result.num_edges,
                            result.num_partitions,
                            result.num_runs,
                            result.best_colors,
                            f"{result.avg_colors:.2f}",
                            f"{result.std_colors:.2f}",
                            f"{result.total_runtime_seconds:.3f}",
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            result.instance_name,
                            result.num_vertices,
                            result.num_edges,
                            result.num_partitions,
                            result.status.value,
                            result.num_colors if result.num_colors is not None else "",
                            f"{result.runtime_seconds:.3f}",
                            f"{result.gap:.4f}" if result.gap is not None else "",
                        ]
                    )

        print(f"\nResults saved to: {filepath}")
        return filepath

    def save_params_json(self, csv_filepath: Path) -> Path:
        """
        Save solver parameters to a JSON file alongside the CSV.

        Args:
            csv_filepath: Path to the CSV file (JSON will be saved with same name)

        Returns:
            Path to the saved JSON file
        """
        json_filepath = csv_filepath.with_suffix(".json")

        if isinstance(self.solver, (ACOSolver, TabuSearchSolver)):
            params = self.solver.get_params()
        else:
            params = {
                "solver": "ILP",
                "backend": self.solver.solver_name,
                "time_limit_seconds": self.solver.time_limit_seconds,
            }

        params["timestamp"] = datetime.now().isoformat()
        params["num_instances"] = len(self.results)

        with open(json_filepath, "w") as f:
            json.dump(params, f, indent=2)

        print(f"Parameters saved to: {json_filepath}")
        return json_filepath

    def print_summary(self):
        """Print a summary of results."""
        if not self.results:
            print("No results to summarize.")
            return

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        total = len(self.results)

        if self._solver_type in ["ACO", "TabuSearch"]:
            heuristic_results = [r for r in self.results if isinstance(r, (ACOResult, TabuSearchResult))]
            avg_best = sum(r.best_colors for r in heuristic_results) / total
            avg_avg = sum(r.avg_colors for r in heuristic_results) / total
            avg_time = sum(r.total_runtime_seconds for r in heuristic_results) / total

            print(f"Total instances: {total}")
            print(f"Runs per instance: {heuristic_results[0].num_runs if heuristic_results else 'N/A'}")
            print(f"Avg best colors: {avg_best:.2f}")
            print(f"Avg avg colors: {avg_avg:.2f}")
            print(f"Avg time per instance: {avg_time:.2f}s")
        else:
            ilp_results = [r for r in self.results if isinstance(r, SolverResult)]
            optimal = sum(1 for r in ilp_results if r.status.value == "optimal")
            feasible = sum(1 for r in ilp_results if r.status.value == "feasible")
            timeout = sum(1 for r in ilp_results if r.status.value == "timeout")
            infeasible = sum(1 for r in ilp_results if r.status.value == "infeasible")

            print(f"Total instances: {total}")
            print(f"  Optimal:    {optimal} ({100 * optimal / total:.1f}%)")
            print(f"  Feasible:   {feasible} ({100 * feasible / total:.1f}%)")
            print(f"  Timeout:    {timeout} ({100 * timeout / total:.1f}%)")
            print(f"  Infeasible: {infeasible} ({100 * infeasible / total:.1f}%)")

            solved = [r for r in ilp_results if r.num_colors is not None]
            if solved:
                avg_colors = sum(r.num_colors for r in solved if r.num_colors is not None) / len(solved)
                avg_time = sum(r.runtime_seconds for r in solved) / len(solved)
                print(f"\nSolved instances: {len(solved)}")
                print(f"  Avg colors: {avg_colors:.2f}")
                print(f"  Avg time:   {avg_time:.2f}s")

    def print_table(self):
        """Print results as a formatted table."""
        if not self.results:
            print("No results to display.")
            return

        if self._solver_type in ["ACO", "TabuSearch"]:
            self._print_heuristic_table()
        else:
            self._print_ilp_table()

    def _print_heuristic_table(self):
        """Print heuristic solver (ACO/TabuSearch) results as a formatted table."""
        print(
            f"\n{'Instance':<25} {'V':>5} {'E':>6} {'P':>4} "
            f"{'Runs':>5} {'Best':>5} {'Avg':>7} {'Std':>6} {'Time':>8}"
        )
        print("-" * 85)

        for r in self.results:
            if isinstance(r, (ACOResult, TabuSearchResult)):
                print(
                    f"{r.instance_name:<25} {r.num_vertices:>5} {r.num_edges:>6} "
                    f"{r.num_partitions:>4} {r.num_runs:>5} {r.best_colors:>5} "
                    f"{r.avg_colors:>7.2f} {r.std_colors:>6.2f} {r.total_runtime_seconds:>7.2f}s"
                )

    def _print_ilp_table(self):
        """Print ILP results as a formatted table."""
        print(f"\n{'Instance':<30} {'V':>6} {'E':>6} {'P':>4} {'Status':<10} {'Colors':>7} {'Time(s)':>10}")
        print("-" * 80)

        for r in self.results:
            if isinstance(r, SolverResult):
                colors_str = str(r.num_colors) if r.num_colors is not None else "-"
                print(
                    f"{r.instance_name:<30} {r.num_vertices:>6} {r.num_edges:>6} "
                    f"{r.num_partitions:>4} {r.status.value:<10} {colors_str:>7} {r.runtime_seconds:>10.3f}"
                )
