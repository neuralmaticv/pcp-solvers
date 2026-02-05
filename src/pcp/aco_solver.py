"""
ACO Solver for the Partition Coloring Problem.

Implements Ant Colony Optimization based on the approach described in:
"An Ant Colony Optimization Algorithm for the Partition Coloring Problem"
by Stefka Fidanova et al.

## Algorithm:
1. Initialize pheromone values tau[v] for each vertex
2. For each iteration:
   a. Each ant constructs a solution:
      - Visit partitions in fixed order (0, 1, 2, ...)
      - Select one vertex from each partition using probability based on
        pheromone tau[v] and heuristic eta[v]
   b. Evaluate each solution using DSATUR coloring
   c. Update global best if improved
   d. Update pheromones: evaporate all, reinforce global best vertices

3. Return the best solution found
"""

import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

from .dsatur import count_colors, dsatur_coloring, verify_coloring
from .instance import PCPInstance


@dataclass
class ACOResult:
    """Result of ACO solver on a PCP instance.

    Supports both single-run and multi-run evaluation.
    For proper statistical evaluation, use num_runs > 1.
    """

    instance_name: str
    num_vertices: int
    num_edges: int
    num_partitions: int
    num_runs: int  # Number of independent runs
    best_colors: int  # Best result across all runs
    avg_colors: float  # Average colors across runs
    std_colors: float  # Standard deviation of colors (0 if single run)
    total_runtime_seconds: float  # Total time for all runs
    all_colors: list[int] = field(default_factory=list)  # Colors from each run
    best_selected_vertices: Optional[list[int]] = None  # Best solution vertices
    best_vertex_colors: Optional[dict[int, int]] = None  # Best solution coloring

    def to_csv_row(self) -> str:
        """Format as CSV row."""
        return ",".join(
            [
                self.instance_name,
                str(self.num_vertices),
                str(self.num_edges),
                str(self.num_partitions),
                str(self.num_runs),
                str(self.best_colors),
                f"{self.avg_colors:.2f}",
                f"{self.std_colors:.2f}",
                f"{self.total_runtime_seconds:.3f}",
            ]
        )

    @staticmethod
    def csv_header() -> str:
        """Return CSV header."""
        return "instance,vertices,edges,partitions,runs,best,avg,std,total_time_s"


class ACOSolver:
    """
    ACO-based solver for the PCP.

    Each ant constructs a solution by selecting one vertex per partition,
    guided by pheromone trails and a conflict-based heuristic.
    Solutions are evaluated using DSATUR graph coloring.
    """

    def __init__(
        self,
        num_ants: Optional[int] = None,  # Default: number of partitions
        num_iterations: int = 100,
        alpha: float = 1.0,  # Pheromone importance
        beta: float = 2.0,  # Heuristic importance
        rho: float = 0.1,  # Evaporation rate
        initial_pheromone: float = 1.0,
        num_runs: int = 1,  # Number of independent runs for evaluation
        base_seed: Optional[int] = None,  # Base seed (seeds will be base_seed + i for each run)
        verbose: bool = False,
    ):
        """
        Initialize the ACO solver.

        Args:
            num_ants: Number of ants per iteration (default: num_partitions)
            num_iterations: Maximum iterations to run
            alpha: Pheromone importance exponent (tau^alpha)
            beta: Heuristic importance exponent (eta^beta)
            rho: Evaporation rate (0 < rho < 1), pheromone decays by (1 - rho)
            initial_pheromone: Starting pheromone value for all vertices
            num_runs: Number of independent runs for statistical evaluation
            base_seed: Base random seed for reproducibility (None for random)
            verbose: Whether to print progress
        """
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.initial_pheromone = initial_pheromone
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.verbose = verbose

    def solve(self, instance: PCPInstance) -> ACOResult:
        """
        Solve a PCP instance using ACO.

        Runs the algorithm num_runs times with different seeds and returns
        aggregated statistics. For a single run, set num_runs=1.

        Args:
            instance: The PCP instance to solve

        Returns:
            ACOResult with statistics across all runs
        """
        import time

        start_time = time.time()

        all_colors: list[int] = []
        best_selection: Optional[list[int]] = None
        best_coloring: Optional[dict[int, int]] = None
        best_num_colors: Optional[int] = None

        for run_idx in range(self.num_runs):
            # Set seed for this run
            if self.base_seed is not None:
                seed = self.base_seed + run_idx
            else:
                seed = None

            num_colors, selection, coloring = self._single_run(instance, seed)

            if num_colors is not None:
                all_colors.append(num_colors)

                # Track best result
                if best_num_colors is None or num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_selection = selection
                    best_coloring = coloring

                    if self.verbose:
                        print(f"  Run {run_idx + 1}/{self.num_runs}: new best = {best_num_colors} colors")

        total_runtime = time.time() - start_time

        # Compute statistics
        if all_colors:
            best_colors = min(all_colors)
            avg_colors = statistics.mean(all_colors)
            std_colors = statistics.stdev(all_colors) if len(all_colors) > 1 else 0.0
        else:
            best_colors = 0
            avg_colors = 0.0
            std_colors = 0.0

        if self.verbose:
            print(f"  Final: best={best_colors}, avg={avg_colors:.2f}, std={std_colors:.2f}")

        return ACOResult(
            instance_name=instance.name,
            num_vertices=instance.num_vertices,
            num_edges=instance.num_edges,
            num_partitions=instance.num_partitions,
            num_runs=self.num_runs,
            best_colors=best_colors,
            avg_colors=avg_colors,
            std_colors=std_colors,
            total_runtime_seconds=total_runtime,
            all_colors=all_colors,
            best_selected_vertices=best_selection,
            best_vertex_colors=best_coloring,
        )

    def _single_run(
        self,
        instance: PCPInstance,
        seed: Optional[int],
    ) -> tuple[Optional[int], Optional[list[int]], Optional[dict[int, int]]]:
        """
        Execute a single ACO run.

        Args:
            instance: PCP instance
            seed: Random seed for this run

        Returns:
            Tuple of (num_colors, selected_vertices, vertex_colors)
        """
        # Initialize random generator
        rng = random.Random(seed)

        # Set number of ants
        num_ants = self.num_ants if self.num_ants is not None else instance.num_partitions

        # Initialize pheromone for each vertex
        pheromone: dict[int, float] = {v: self.initial_pheromone for v in range(instance.num_vertices)}

        # Track global best for this run
        best_selection: Optional[list[int]] = None
        best_coloring: Optional[dict[int, int]] = None
        best_num_colors: Optional[int] = None

        # Main ACO loop
        for _ in range(self.num_iterations):
            # Each ant constructs a solution
            for _ in range(num_ants):
                selection = self._construct_solution(instance, pheromone, rng)

                # Evaluate using DSATUR
                coloring = dsatur_coloring(selection, instance.adjacency)
                num_colors = count_colors(coloring)

                # Update global best
                if best_num_colors is None or num_colors < best_num_colors:
                    best_num_colors = num_colors
                    best_selection = selection
                    best_coloring = coloring

            # Pheromone update
            self._update_pheromones(pheromone, best_selection, best_num_colors, instance)

        return best_num_colors, best_selection, best_coloring

    def _construct_solution(
        self,
        instance: PCPInstance,
        pheromone: dict[int, float],
        rng: random.Random,
    ) -> list[int]:
        """
        Construct a solution by selecting one vertex per partition.

        Uses fixed partition order (0, 1, 2, ...).

        Args:
            instance: PCP instance
            pheromone: Current pheromone values
            rng: Random number generator

        Returns:
            List of selected vertices (one per partition, in partition order)
        """
        selected: list[int] = []
        selected_set: set[int] = set()

        # Visit partitions in fixed order
        for p_idx in range(instance.num_partitions):
            partition = instance.partitions[p_idx]

            # Compute selection probabilities for vertices in this partition
            probabilities = []
            for v in partition:
                tau = pheromone[v] ** self.alpha
                eta = self._compute_heuristic(v, selected_set, instance) ** self.beta
                probabilities.append(tau * eta)

            # Normalize to get probability distribution
            total = sum(probabilities)
            if total > 0:
                probabilities = [p / total for p in probabilities]
            else:
                # Fallback to uniform if all probabilities are zero
                probabilities = [1.0 / len(partition)] * len(partition)

            # Select vertex using roulette wheel
            selected_vertex = self._roulette_select(partition, probabilities, rng)
            selected.append(selected_vertex)
            selected_set.add(selected_vertex)

        return selected

    def _compute_heuristic(
        self,
        vertex: int,
        selected_set: set[int],
        instance: PCPInstance,
    ) -> float:
        """
        Compute heuristic value for a vertex.

        Dynamic heuristic based on conflicts with already-selected vertices.
        eta[v] = 1 / (1 + conflicts_with_selected[v])

        Args:
            vertex: Vertex to evaluate
            selected_set: Set of already-selected vertices
            instance: PCP instance

        Returns:
            Heuristic value (higher = more attractive)
        """
        # Count conflicts: neighbors that are already selected
        conflicts = len(instance.adjacency[vertex] & selected_set)
        return 1.0 / (1.0 + conflicts)

    def _roulette_select(
        self,
        candidates: list[int],
        probabilities: list[float],
        rng: random.Random,
    ) -> int:
        """
        Select an item using roulette wheel selection.

        Args:
            candidates: List of candidate items
            probabilities: Selection probabilities
            rng: Random number generator

        Returns:
            Selected item
        """
        r = rng.random()
        cumulative = 0.0
        for item, prob in zip(candidates, probabilities):
            cumulative += prob
            if r <= cumulative:
                return item
        # Fallback (should not happen - just in case of rounding errors...)
        return candidates[-1]

    def _update_pheromones(
        self,
        pheromone: dict[int, float],
        best_selection: Optional[list[int]],
        best_num_colors: Optional[int],
        instance: PCPInstance,
    ) -> None:
        """
        Update pheromone values.

        Global best reinforcement only.
        - Evaporate all pheromones by factor (1 - rho)
        - Deposit pheromone on vertices in global best solution

        Args:
            pheromone: Pheromone dict to update (modified in place)
            best_selection: Vertices in global best solution
            best_num_colors: Number of colors in global best
            instance: PCP instance
        """
        # Evaporation: tau[v] = (1 - rho) * tau[v]
        for v in pheromone:
            pheromone[v] *= 1.0 - self.rho

        # Reinforcement on global best vertices
        if best_selection is not None and best_num_colors is not None:
            # Deposit amount inversely proportional to number of colors
            # Better solutions (fewer colors) get more pheromone
            deposit = 1.0 / best_num_colors

            for v in best_selection:
                pheromone[v] += deposit

    def get_params(self) -> dict:
        """
        Get solver parameters as a dictionary.

        Useful for saving configuration to JSON.

        Returns:
            Dictionary of solver parameters
        """
        return {
            "solver": "ACO",
            "num_ants": self.num_ants if self.num_ants is not None else "num_partitions",
            "num_iterations": self.num_iterations,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "initial_pheromone": self.initial_pheromone,
            "num_runs": self.num_runs,
            "base_seed": self.base_seed,
        }

    def verify_solution(self, instance: PCPInstance, result: ACOResult) -> bool:
        """
        Verify that the best solution is valid.

        Args:
            instance: The PCP instance
            result: The ACO result to verify

        Returns:
            True if the best solution is valid
        """
        if result.best_selected_vertices is None or result.best_vertex_colors is None:
            return False

        # Check: exactly one vertex per partition
        if len(result.best_selected_vertices) != instance.num_partitions:
            return False

        selected_set = set(result.best_selected_vertices)
        for partition in instance.partitions:
            count = sum(1 for v in partition if v in selected_set)
            if count != 1:
                return False

        # Check coloring validity
        return verify_coloring(
            result.best_selected_vertices,
            instance.adjacency,
            result.best_vertex_colors,
        )
