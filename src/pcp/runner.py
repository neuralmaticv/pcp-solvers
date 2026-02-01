"""
Experiment runner for PCP instances.

Runs solver on specified instances and collects results.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path

from .ilp_solver import ILPSolver, SolverResult
from .instance import PCPInstance


class ExperimentRunner:
    """Runs experiments on PCP instances and collects results."""

    def __init__(
        self,
        solver: ILPSolver,
        output_dir: Path | None = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            solver: The PCP solver to use
            output_dir: Directory for output files (default: current directory)
        """
        self.solver = solver
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.results: list[SolverResult] = []

    def run_instance(self, filepath: Path) -> SolverResult:
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
    ) -> list[SolverResult]:
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
                print(f"{result.status.value} - {result.num_colors} colors in {result.runtime_seconds:.2f}s")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            results.append(result)

        return results

    def run_all_families(
        self,
        instances_dir: Path,
        families: list[str] | None = None,
        max_per_family: int | None = None,
        from_end: bool = False,
    ) -> dict[str, list[SolverResult]]:
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
            solver_part = f"{solver_name}_" if solver_name else ""
            filename = f"results_{solver_part}{timestamp}.csv"

        filepath = self.output_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "instance",
                    "vertices",
                    "edges",
                    "partitions",
                    "status",
                    "colors",
                    "runtime_s",
                    "gap",
                ]
            )

            for result in self.results:
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

    def print_summary(self):
        """Print a summary of results."""
        if not self.results:
            print("No results to summarize.")
            return

        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")

        total = len(self.results)
        optimal = sum(1 for r in self.results if r.status.value == "optimal")
        feasible = sum(1 for r in self.results if r.status.value == "feasible")
        timeout = sum(1 for r in self.results if r.status.value == "timeout")
        infeasible = sum(1 for r in self.results if r.status.value == "infeasible")

        print(f"Total instances: {total}")
        print(f"  Optimal:    {optimal} ({100 * optimal / total:.1f}%)")
        print(f"  Feasible:   {feasible} ({100 * feasible / total:.1f}%)")
        print(f"  Timeout:    {timeout} ({100 * timeout / total:.1f}%)")
        print(f"  Infeasible: {infeasible} ({100 * infeasible / total:.1f}%)")

        solved = [r for r in self.results if r.num_colors is not None]
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

        # Header
        print(f"\n{'Instance':<30} {'V':>6} {'E':>6} {'P':>4} {'Status':<10} {'Colors':>7} {'Time(s)':>10}")
        print("-" * 80)

        for r in self.results:
            colors_str = str(r.num_colors) if r.num_colors is not None else "-"
            print(
                f"{r.instance_name:<30} {r.num_vertices:>6} {r.num_edges:>6} "
                f"{r.num_partitions:>4} {r.status.value:<10} {colors_str:>7} {r.runtime_seconds:>10.3f}"
            )
