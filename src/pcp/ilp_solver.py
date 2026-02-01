from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ortools.linear_solver import pywraplp

from .instance import PCPInstance


class SolverStatus(Enum):
    """Status of the solver after optimization."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class SolverResult:
    """Result of solving a PCP instance."""

    instance_name: str
    num_vertices: int
    num_edges: int
    num_partitions: int
    status: SolverStatus
    num_colors: Optional[int]  # Optimal/best found chromatic number
    runtime_seconds: float
    selected_vertices: Optional[list[int]]  # Which vertex selected from each partition
    vertex_colors: Optional[dict[int, int]]  # Color assignment for selected vertices
    gap: Optional[float]  # MIP gap if not optimal

    def to_csv_row(self) -> str:
        """Format as CSV row."""
        return ",".join(
            [
                self.instance_name,
                str(self.num_vertices),
                str(self.num_edges),
                str(self.num_partitions),
                self.status.value,
                str(self.num_colors) if self.num_colors is not None else "",
                f"{self.runtime_seconds:.3f}",
                f"{self.gap:.4f}" if self.gap is not None else "",
            ]
        )

    @staticmethod
    def csv_header() -> str:
        """Return CSV header."""
        return "instance,vertices,edges,partitions,status,colors,runtime_s,gap"


class ILPSolver:
    """
    ILP-based solver for the Partition Coloring Problem.

    Uses OR-Tools with SCIP backend by default.
    """

    def __init__(
        self,
        time_limit_seconds: float = 300.0,
        solver_name: str = "SCIP",
        verbose: bool = False,
    ):
        """
        Initialize the solver.

        Args:
            time_limit_seconds: Maximum solving time
            solver_name: Backend solver ("SCIP", "CBC", "GLOP" for LP relaxation)
            verbose: Whether to print solver output
        """
        self.time_limit_seconds = time_limit_seconds
        self.solver_name = solver_name
        self.verbose = verbose

    def solve(self, instance: PCPInstance) -> SolverResult:
        """
        Solve a PCP instance using ILP.

        Args:
            instance: The PCP instance to solve

        Returns:
            SolverResult with solution details
        """
        # Create solver
        solver = pywraplp.Solver.CreateSolver(self.solver_name)
        solver.SetTimeLimit(int(self.time_limit_seconds * 1000))

        # Upper bound on colors: at most k partitions => at most k colors needed
        max_colors = instance.num_partitions

        # Decision variables
        # y[v,c] = 1 if vertex v is selected and assigned color c
        y = {}
        for v in range(instance.num_vertices):
            for c in range(max_colors):
                y[v, c] = solver.BoolVar(f"y_{v}_{c}")

        # w[c] = 1 if color c is used
        w = {}
        for c in range(max_colors):
            w[c] = solver.BoolVar(f"w_{c}")

        # Constraint 1: exactly one vertex-color pair per partition
        # This combines partition selection and color assignment
        for p_idx, partition in enumerate(instance.partitions):
            solver.Add(
                sum(y[v, c] for v in partition for c in range(max_colors)) == 1,
                f"partition_{p_idx}",
            )

        # Constraint 3: adjacent vertices cannot share a color
        for u, v in instance.edges:
            for c in range(max_colors):
                solver.Add(y[u, c] + y[v, c] <= 1, f"conflict_{u}_{v}_{c}")

        # Constraint 4: color usage tracking
        for v in range(instance.num_vertices):
            for c in range(max_colors):
                solver.Add(y[v, c] <= w[c], f"color_used_{v}_{c}")

        # Symmetry breaking: use colors in order
        # If color c+1 is used, then color c must be used
        for c in range(max_colors - 1):
            solver.Add(w[c] >= w[c + 1], f"symmetry_{c}")

        # Objective: minimize number of colors used
        solver.Minimize(sum(w[c] for c in range(max_colors)))

        if self.verbose:
            print(f"Solving {instance.name}...")
            print(f"  Variables: {solver.NumVariables()}")
            print(f"  Constraints: {solver.NumConstraints()}")

        # Run solver
        status = solver.Solve()

        # Process result
        runtime = solver.WallTime() / 1000.0  # convert ms to seconds

        if status == pywraplp.Solver.OPTIMAL:
            result_status = SolverStatus.OPTIMAL
            gap = 0.0
        elif status == pywraplp.Solver.FEASIBLE:
            result_status = SolverStatus.FEASIBLE
            # compute gap if possible
            if solver.Objective().BestBound() > 0:
                gap = abs(solver.Objective().Value() - solver.Objective().BestBound()) / solver.Objective().Value()
            else:
                gap = None
        elif status == pywraplp.Solver.INFEASIBLE:
            result_status = SolverStatus.INFEASIBLE
            gap = None
        else:
            result_status = SolverStatus.TIMEOUT
            gap = None

        # extract solution if found
        num_colors = None
        selected_vertices = None
        vertex_colors = None

        if result_status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            num_colors = int(round(solver.Objective().Value()))

            # Find selected vertices and their colors from y[v,c]
            selected_vertices = []
            vertex_colors = {}
            for partition in instance.partitions:
                for v in partition:
                    for c in range(max_colors):
                        if y[v, c].solution_value() > 0.5:
                            selected_vertices.append(v)
                            vertex_colors[v] = c
                            break
                    else:
                        continue
                    break  # Found vertex for this partition

            if self.verbose:
                print(f"  Solution found: {num_colors} colors")
                print(f"  Selected vertices: {selected_vertices}")
                print(f"  Colors: {vertex_colors}")

        return SolverResult(
            instance_name=instance.name,
            num_vertices=instance.num_vertices,
            num_edges=instance.num_edges,
            num_partitions=instance.num_partitions,
            status=result_status,
            num_colors=num_colors,
            runtime_seconds=runtime,
            selected_vertices=selected_vertices,
            vertex_colors=vertex_colors,
            gap=gap,
        )

    def verify_solution(self, instance: PCPInstance, result: SolverResult) -> bool:
        """
        Verify that a solution is valid.

        Args:
            instance: The PCP instance
            result: The solver result to verify

        Returns:
            True if the solution is valid
        """
        if result.selected_vertices is None or result.vertex_colors is None:
            return False

        # Check: exactly one vertex per partition
        selected_set = set(result.selected_vertices)
        for partition in instance.partitions:
            count = sum(1 for v in partition if v in selected_set)
            if count != 1:
                return False

        # Check: no adjacent vertices have the same color
        for u, v in instance.edges:
            if u in selected_set and v in selected_set:
                if result.vertex_colors.get(u) == result.vertex_colors.get(v):
                    return False

        return True
