"""
Tabu Search for the Partition Coloring Problem (TS-PCP).

This module implements the algorithms from:
"Routing and wavelength assignment by partition coloring"

The implementation includes:
- onestepCD: Construction heuristic
- TS-PCP: Tabu search for color reduction

Algorithm overview:
1. Build initial solution using onestepCD with maxC colors
2. Iteratively attempt to reduce colors using tabu search:
   - Randomly recolor nodes to use maxC-1 colors
   - Search for feasible solution (zero conflicts) via neighborhood exploration
   - If successful, accept and continue reduction
   - Otherwise, stop and return best solution found
"""

import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

from .instance import PCPInstance


@dataclass
class TabuSearchResult:
    """Result of Tabu Search solver on a PCP instance."""

    instance_name: str
    num_vertices: int
    num_edges: int
    num_partitions: int
    num_runs: int
    best_colors: int
    avg_colors: float
    std_colors: float
    total_runtime_seconds: float
    all_colors: list[int] = field(default_factory=list)
    best_selected_vertices: Optional[dict[int, int]] = None
    best_vertex_colors: Optional[dict[int, int]] = None

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


class TabuSearchSolver:
    """
    Tabu Search solver for the Partition Coloring Problem.

    Uses onestepCD construction heuristic followed by iterative color reduction
    via tabu search on a 1-opt neighborhood.
    """

    def __init__(
        self,
        tabu_tenure_range: tuple[float, float] = (0.0, 0.5),  # As fraction of maxC-1
        max_iter_factor: int = 5,  # maxIter = q * (maxC-1) * Fend
        num_runs: int = 1,
        base_seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize the Tabu Search solver.

        Args:
            tabu_tenure_range: Range for random tabu tenure as (min_frac, max_frac)
                              Tenure = uniform[min_frac * C, max_frac * C] where C = maxC-1
            max_iter_factor: Factor Fend for computing maxIter = q * (maxC-1) * Fend
            num_runs: Number of independent runs for statistical evaluation
            base_seed: Base random seed for reproducibility
            verbose: Whether to print progress
        """
        self.tabu_tenure_range = tabu_tenure_range
        self.max_iter_factor = max_iter_factor
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.verbose = verbose

    def solve(self, instance: PCPInstance) -> TabuSearchResult:
        """
        Solve a PCP instance using Tabu Search.

        Args:
            instance: The PCP instance to solve

        Returns:
            TabuSearchResult with statistics across all runs
        """
        import time

        start_time = time.time()

        all_colors: list[int] = []
        best_selection: Optional[dict[int, int]] = None
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

        return TabuSearchResult(
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
    ) -> tuple[Optional[int], Optional[dict[int, int]], Optional[dict[int, int]]]:
        """Execute a single run."""
        rng = random.Random(seed)

        # Step 1: Get initial solution using onestepCD
        initial_selection, initial_coloring = self._onestepCD(instance, rng)
        max_colors = max(initial_coloring.values()) + 1

        if self.verbose:
            print(f"    Initial solution: {max_colors} colors")

        # Step 2: Try to reduce colors using tabu search
        best_selection = initial_selection
        best_coloring = initial_coloring
        best_num_colors = max_colors

        current_max_colors = max_colors
        while current_max_colors > 1:
            # Try to find a feasible solution with current_max_colors - 1 colors
            result = self._ts_pcp(instance, best_selection, best_coloring, current_max_colors, rng)

            if result is not None:
                # Found feasible solution with fewer colors
                selection, coloring = result
                num_colors = max(coloring.values()) + 1
                best_selection = selection
                best_coloring = coloring
                best_num_colors = num_colors
                current_max_colors = num_colors

                if self.verbose:
                    print(f"    Reduced to {num_colors} colors")
            else:
                # Could not reduce further
                if self.verbose:
                    print(f"    Could not reduce below {current_max_colors} colors")
                break

        return best_num_colors, best_selection, best_coloring

    def _onestepCD(
        self,
        instance: PCPInstance,
        rng: random.Random,
    ) -> tuple[dict[int, int], dict[int, int]]:
        """
        Construction heuristic: One Step Color Degree (onestepCD).

        Args:
            instance: PCP instance
            rng: Random number generator

        Returns:
            Tuple of (selected_vertices, coloring)
            - selected_vertices: Dict mapping partition index to selected vertex
            - coloring: Dict mapping selected vertices to colors
        """
        # Build induced graph (remove edges within same partition)
        induced_adj = self._build_induced_adjacency(instance)

        selected: dict[int, int] = {}  # partition_idx -> vertex (V' in paper)
        coloring: dict[int, int] = {}  # color assignment
        remaining_partitions = list(range(instance.num_partitions))

        while remaining_partitions:
            # For each uncolored partition, find node with minimum CD
            X = []  # Candidates with min CD from each partition
            for p_idx in remaining_partitions:
                partition = instance.partitions[p_idx]
                min_cd = float("inf")
                min_cd_vertex = None

                for v in partition:
                    cd = self._compute_color_degree(v, induced_adj, selected, coloring)
                    if cd < min_cd:
                        min_cd = cd
                        min_cd_vertex = v

                if min_cd_vertex is not None:
                    X.append((min_cd_vertex, min_cd))

            # Among candidates, select vertex with maximum CD
            if not X:
                break

            x, _ = max(X, key=lambda item: item[1])

            # Color x with minimum possible color
            used_colors = set()
            for neighbor in induced_adj[x]:
                if neighbor in coloring:
                    used_colors.add(coloring[neighbor])

            color = 0
            while color in used_colors:
                color += 1

            # Assign color
            partition_idx = instance.vertex_to_partition[x]
            selected[partition_idx] = x
            coloring[x] = color

            # Remove partition containing x from remaining
            remaining_partitions.remove(partition_idx)

        return selected, coloring

    def _compute_color_degree(
        self,
        vertex: int,
        adjacency: dict[int, set[int]],
        selected: dict[int, int],
        coloring: dict[int, int],
    ) -> int:
        """
        Compute Color Degree (saturation degree) of a vertex.

        CD(v) = number of different colors used by neighbors of v in selected vertices.

        Args:
            vertex: Vertex to evaluate
            adjacency: Adjacency dict
            selected: Dict mapping partition to selected vertex
            coloring: Current color assignment

        Returns:
            Color saturation degree
        """
        neighbor_colors = set()
        for neighbor in adjacency[vertex]:
            if neighbor in coloring:
                neighbor_colors.add(coloring[neighbor])
        return len(neighbor_colors)

    def _ts_pcp(
        self,
        instance: PCPInstance,
        prev_selection: dict[int, int],
        prev_coloring: dict[int, int],
        max_colors: int,
        rng: random.Random,
    ) -> Optional[tuple[dict[int, int], dict[int, int]]]:
        """
        Tabu Search for PCP - attempts to find feasible solution with max_colors - 1 colors.

        Args:
            instance: PCP instance
            prev_selection: Previous selected vertices (one per partition)
            prev_coloring: Previous coloring
            max_colors: Current number of colors
            rng: Random number generator

        Returns:
            (selection, coloring) if feasible solution found, None otherwise
        """
        target_colors = max_colors - 1
        induced_adj = self._build_induced_adjacency(instance)

        # Build initial solution S' by randomly recoloring nodes with color maxC
        S_prime_selection = prev_selection.copy()
        S_prime_coloring = {}
        for partition_idx, vertex in S_prime_selection.items():
            old_color = prev_coloring[vertex]
            if old_color >= target_colors:
                new_color = rng.randint(0, target_colors - 1)
            else:
                new_color = old_color
            S_prime_coloring[vertex] = new_color

        # Free tabu list and set iter = 0
        tabu_list: dict[tuple[int, int], int] = {}
        iteration = 0

        # Compute max iterations: q * (maxC-1) * Fend
        q = instance.num_partitions
        C = target_colors
        max_iter = q * C * self.max_iter_factor

        # Maintain a cached set of selected vertices for O(1) lookups
        selected_set = set(S_prime_selection.values())

        # Compute initial conflict count once
        current_conflicts = self._count_conflicts_fast(
            S_prime_selection, S_prime_coloring, induced_adj, instance
        )

        if current_conflicts == 0:
            return S_prime_selection, S_prime_coloring

        # Let Q be the set of components with conflicts in S'
        Q = self._get_conflict_components(S_prime_selection, S_prime_coloring, induced_adj, instance)

        # Main loop: while Q ≠ ∅
        while Q and iteration < max_iter:
            # Randomly select k ∈ Q and update Q ← Q \ {k}
            k = rng.choice(list(Q))
            Q.discard(k)

            # Set reduction ← FALSE
            reduction = False

            # Set maxConflicts ← ∞
            best_tentative_conflicts = float("inf")
            best_i_bar = None
            best_ell_bar = None

            partition_k = instance.partitions[k]
            current_vertex_k = S_prime_selection[k]
            current_color_k = S_prime_coloring.get(current_vertex_k)

            # Precompute: conflicts contributed by the current vertex in partition k
            old_vertex_conflicts = self._count_vertex_conflicts(
                current_vertex_k, current_color_k, selected_set, S_prime_coloring, induced_adj
            ) if current_color_k is not None else 0

            # For each i ∈ Vk and for each color ℓ while .NOT.reduction
            for i in partition_k:
                if reduction:
                    break
                for ell in range(target_colors):
                    if reduction:
                        break

                    # Check if pair (i,ℓ) is tabu
                    is_tabu = (i, ell) in tabu_list and tabu_list[(i, ell)] > iteration

                    # Compute tentative conflicts incrementally:
                    # Remove old vertex's conflicts, add new vertex's conflicts
                    if i == current_vertex_k:
                        # Same vertex, just recolor
                        new_vertex_conflicts = self._count_vertex_conflicts(
                            i, ell, selected_set, S_prime_coloring, induced_adj
                        )
                    else:
                        # Different vertex: temporarily update selected_set
                        selected_set.discard(current_vertex_k)
                        selected_set.add(i)
                        # Temporarily update coloring for the new vertex
                        S_prime_coloring[i] = ell
                        old_entry = S_prime_coloring.pop(current_vertex_k, None)

                        new_vertex_conflicts = self._count_vertex_conflicts(
                            i, ell, selected_set, S_prime_coloring, induced_adj
                        )

                        # Restore state
                        if old_entry is not None:
                            S_prime_coloring[current_vertex_k] = old_entry
                        del S_prime_coloring[i]
                        selected_set.discard(i)
                        selected_set.add(current_vertex_k)

                    tentative_conflicts = current_conflicts - old_vertex_conflicts + new_vertex_conflicts

                    # Aspiration criterion
                    aspiration = tentative_conflicts < current_conflicts

                    if not is_tabu or aspiration:
                        # Track best neighbor
                        if tentative_conflicts < best_tentative_conflicts:
                            best_tentative_conflicts = tentative_conflicts
                            best_i_bar = i
                            best_ell_bar = ell

                        # If improvement over current S', apply and set reduction
                        if tentative_conflicts < current_conflicts:
                            # Apply the move
                            if i != current_vertex_k:
                                selected_set.discard(current_vertex_k)
                                selected_set.add(i)
                                S_prime_selection[k] = i
                                if current_vertex_k in S_prime_coloring:
                                    del S_prime_coloring[current_vertex_k]
                            S_prime_coloring[i] = ell
                            current_conflicts = tentative_conflicts

                            reduction = True
                            iteration += 1
                            if current_conflicts == 0:
                                return S_prime_selection, S_prime_coloring
                            Q = self._get_conflict_components(
                                S_prime_selection, S_prime_coloring, induced_adj, instance
                            )

            if current_conflicts == 0:
                return S_prime_selection, S_prime_coloring

            # Move to best neighbor if no improvement found
            if not reduction and best_i_bar is not None and best_ell_bar is not None:
                # Insert pair in tabu list for TabuTenure iterations
                tenure = self._compute_tabu_tenure(C, rng)
                tabu_list[(best_i_bar, best_ell_bar)] = iteration + tenure

                # Apply the best move
                if best_i_bar != current_vertex_k:
                    selected_set.discard(current_vertex_k)
                    selected_set.add(best_i_bar)
                    S_prime_selection[k] = best_i_bar
                    if current_vertex_k in S_prime_coloring:
                        del S_prime_coloring[current_vertex_k]
                S_prime_coloring[best_i_bar] = best_ell_bar
                current_conflicts = best_tentative_conflicts

                iteration += 1

                if current_conflicts == 0:
                    return S_prime_selection, S_prime_coloring

                if iteration < max_iter:
                    Q = self._get_conflict_components(
                        S_prime_selection, S_prime_coloring, induced_adj, instance
                    )

        if current_conflicts == 0:
            return S_prime_selection, S_prime_coloring

        return None

    def _build_induced_adjacency(self, instance: PCPInstance) -> dict[int, set[int]]:
        """
        Build induced adjacency by removing edges within same partition.

        Args:
            instance: PCP instance

        Returns:
            Adjacency dict with intra-partition edges removed
        """
        induced_adj = {}
        for v in range(instance.num_vertices):
            induced_adj[v] = set()
            v_partition = instance.vertex_to_partition[v]
            for neighbor in instance.adjacency[v]:
                neighbor_partition = instance.vertex_to_partition[neighbor]
                if v_partition != neighbor_partition:
                    induced_adj[v].add(neighbor)
        return induced_adj

    def _count_vertex_conflicts(
        self,
        vertex: int,
        color: int,
        selected_vertices: set[int],
        coloring: dict[int, int],
        adjacency: dict[int, set[int]],
    ) -> int:
        """Count conflicts for a single vertex against selected neighbors."""
        conflicts = 0
        for neighbor in adjacency.get(vertex, set()):
            if neighbor in selected_vertices and coloring.get(neighbor) == color:
                conflicts += 1
        return conflicts

    def _count_conflicts_fast(
        self,
        selection: dict[int, int],
        coloring: dict[int, int],
        adjacency: dict[int, set[int]],
        instance: PCPInstance,
    ) -> int:
        """
        Efficiently count coloring conflicts in current solution.

        A conflict is a pair of adjacent vertices in different partitions
        that have the same color. This implementation minimizes redundant lookups.

        Args:
            selection: Dict mapping partition to selected vertex
            coloring: Color assignment
            adjacency: Adjacency dict (induced, no intra-partition edges)
            instance: PCP instance

        Returns:
            Number of conflicts
        """
        conflicts = 0
        selected_vertices = set(selection.values())

        for v in selection.values():
            v_color = coloring.get(v)
            if v_color is None:
                continue

            # Count conflicts with neighbors
            for neighbor in adjacency.get(v, set()):
                if neighbor in selected_vertices:
                    neighbor_color = coloring.get(neighbor)
                    if neighbor_color == v_color and v < neighbor:
                        # Only count each conflict once (v < neighbor ensures this)
                        conflicts += 1

        return conflicts

    def _get_conflict_components(
        self,
        selection: dict[int, int],
        coloring: dict[int, int],
        adjacency: dict[int, set[int]],
        instance: PCPInstance,
    ) -> set[int]:
        """
        Get set of partition indices involved in conflicts.

        Args:
            selection: Dict mapping partition to selected vertex
            coloring: Color assignment
            adjacency: Adjacency dict (induced, no intra-partition edges)
            instance: PCP instance

        Returns:
            Set of partition indices with conflicts
        """
        conflict_components = set()
        selected_vertices = set(selection.values())

        for v in selection.values():
            v_color = coloring.get(v)
            if v_color is None:
                continue

            for neighbor in adjacency.get(v, set()):
                if neighbor in selected_vertices:
                    neighbor_color = coloring.get(neighbor)
                    if neighbor_color == v_color:
                        # Both partitions are in conflict
                        v_partition = instance.vertex_to_partition[v]
                        neighbor_partition = instance.vertex_to_partition[neighbor]
                        conflict_components.add(v_partition)
                        conflict_components.add(neighbor_partition)

        return conflict_components

    def _compute_tabu_tenure(self, C: int, rng: random.Random) -> int:
        """
        Compute tabu tenure based on current number of colors.

        Args:
            C: Current number of colors (maxC - 1)
            rng: Random number generator

        Returns:
            Tabu tenure (number of iterations)
        """
        min_frac, max_frac = self.tabu_tenure_range
        min_tenure = int(min_frac * C)
        max_tenure = int(max_frac * C)
        if max_tenure < min_tenure:
            max_tenure = min_tenure
        if max_tenure == 0:
            return 1
        return rng.randint(min_tenure, max_tenure)

    def get_params(self) -> dict:
        """Get solver parameters as a dictionary."""
        return {
            "solver": "TabuSearch",
            "tabu_tenure_range": self.tabu_tenure_range,
            "max_iter_factor": self.max_iter_factor,
            "num_runs": self.num_runs,
            "base_seed": self.base_seed,
        }

    def verify_solution(self, instance: PCPInstance, result: TabuSearchResult) -> bool:
        """
        Verify that the best solution is valid.

        Args:
            instance: The PCP instance
            result: The result to verify

        Returns:
            True if the best solution is valid
        """
        if result.best_selected_vertices is None or result.best_vertex_colors is None:
            return False

        # Check: exactly one vertex per partition
        if len(result.best_selected_vertices) != instance.num_partitions:
            return False

        # Check: each partition has exactly one selected vertex
        selected_set = set(result.best_selected_vertices.values())
        if len(selected_set) != instance.num_partitions:
            return False

        # Check coloring validity (no conflicts)
        induced_adj = self._build_induced_adjacency(instance)
        conflicts = self._count_conflicts_fast(
            result.best_selected_vertices,
            result.best_vertex_colors,
            induced_adj,
            instance,
        )

        return conflicts == 0
