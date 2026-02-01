"""
PCP Instance data structure and parser.

Instance file format:
- Line 1: n (number of vertices, numbered 0 to n-1)
- Line 2: m (number of edges)
- Line 3: p (number of partitions)
- Next m lines: edges as "u v" pairs
- Next p lines: partition definitions (space-separated vertex lists)
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PCPInstance:
    """Represents a Partition Coloring Problem instance."""

    name: str
    num_vertices: int
    num_edges: int
    num_partitions: int
    edges: list[tuple[int, int]]
    partitions: list[list[int]]

    # Derived data, computed after loading
    adjacency: dict[int, set[int]] = field(default_factory=dict, repr=False)
    vertex_to_partition: dict[int, int] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        """Compute derived data structures."""
        # Build adjacency list
        self.adjacency = {v: set() for v in range(self.num_vertices)}
        for u, v in self.edges:
            self.adjacency[u].add(v)
            self.adjacency[v].add(u)

        # Map each vertex to its partition index
        self.vertex_to_partition = {}
        for p_idx, partition in enumerate(self.partitions):
            for v in partition:
                self.vertex_to_partition[v] = p_idx

    @classmethod
    def from_file(cls, filepath: str | Path) -> "PCPInstance":
        """
        Parse a PCP instance from a file.

        Args:
            filepath: Path to the instance file

        Returns:
            PCPInstance object

        Raises:
            ValueError: If the file format is invalid
        """
        filepath = Path(filepath)

        with open(filepath, "r") as f:
            lines = [line.strip() for line in f.readlines()]

        # Remove empty lines at the end
        while lines and not lines[-1]:
            lines.pop()

        if len(lines) < 3:
            raise ValueError(f"Invalid instance file: {filepath} - too few lines")

        try:
            n = int(lines[0])  # vertices
            m = int(lines[1])  # edges
            p = int(lines[2])  # partitions
        except ValueError as e:
            raise ValueError(f"Invalid header in {filepath}: {e}")

        # Parse edges (lines 3 to 3+m-1, i.e., indices 3 to 3+m)
        edges = []
        edge_start = 3
        for i in range(edge_start, edge_start + m):
            if i >= len(lines):
                raise ValueError(f"Missing edge on line {i + 1} in {filepath}")
            parts = lines[i].split()
            if len(parts) < 2:
                raise ValueError(f"Invalid edge format on line {i + 1} in {filepath}")
            u, v = int(parts[0]), int(parts[1])
            edges.append((u, v))

        # Parse partitions (lines 3+m to 3+m+p-1)
        partitions = []
        partition_start = edge_start + m
        for i in range(partition_start, partition_start + p):
            if i >= len(lines):
                raise ValueError(f"Missing partition on line {i + 1} in {filepath}")
            parts = lines[i].split()
            if not parts:
                raise ValueError(f"Empty partition on line {i + 1} in {filepath}")
            partition = [int(v) for v in parts]
            partitions.append(partition)

        # Validate the instance
        cls._validate(n, m, p, edges, partitions, filepath)

        return cls(
            name=filepath.stem,
            num_vertices=n,
            num_edges=m,
            num_partitions=p,
            edges=edges,
            partitions=partitions,
        )

    @staticmethod
    def _validate(n: int, m: int, p: int, edges: list, partitions: list, filepath: Path):
        """Validate instance consistency."""
        # Check edge count
        if len(edges) != m:
            raise ValueError(f"Edge count mismatch in {filepath}: expected {m}, got {len(edges)}")

        # Check partition count
        if len(partitions) != p:
            raise ValueError(f"Partition count mismatch in {filepath}: expected {p}, got {len(partitions)}")

        # Check that all vertices in edges are valid
        for u, v in edges:
            if not (0 <= u < n and 0 <= v < n):
                raise ValueError(f"Invalid vertex in edge ({u}, {v}) in {filepath}")

        # Check that partitions cover all vertices exactly once
        all_vertices = set()
        for partition in partitions:
            for v in partition:
                if not (0 <= v < n):
                    raise ValueError(f"Invalid vertex {v} in partition in {filepath}")
                if v in all_vertices:
                    raise ValueError(f"Vertex {v} appears in multiple partitions in {filepath}")
                all_vertices.add(v)

        if all_vertices != set(range(n)):
            missing = set(range(n)) - all_vertices
            raise ValueError(f"Vertices {missing} not in any partition in {filepath}")

    def get_edges_between_partitions(self, p1: int, p2: int) -> list[tuple[int, int]]:
        """Get all edges between two partitions."""
        v1 = set(self.partitions[p1])
        v2 = set(self.partitions[p2])
        return [(u, v) for u, v in self.edges if (u in v1 and v in v2) or (u in v2 and v in v1)]

    def __str__(self) -> str:
        return f"PCPInstance({self.name}: n={self.num_vertices}, m={self.num_edges}, p={self.num_partitions})"
