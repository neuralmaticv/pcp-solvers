"""
DSATUR (Degree of Saturation) Graph Coloring Algorithm.

DSATUR is a greedy graph coloring heuristic that selects vertices based on
their "saturation degree" - the number of distinct colors already assigned
to their neighbors.

## Algorithm:
1. Start with all vertices uncolored
2. Repeat until all vertices are colored:
   a. Select the uncolored vertex with the highest saturation degree
   b. Break ties by selecting the vertex with highest degree among uncolored vertices
   c. Assign the smallest color not used by any neighbor
3. Return the coloring

Ref: https://www.geeksforgeeks.org/dsa/dsatur-algorithm-for-graph-coloring/
"""

from typing import Optional


def dsatur_coloring(
    vertices: list[int],
    adjacency: dict[int, set[int]],
) -> dict[int, int]:
    """
    Color a graph using the DSATUR algorithm.

    Args:
        vertices: List of vertex IDs to color (the induced subgraph)
        adjacency: Full adjacency dict (vertex -> set of neighbors)
                   Note: neighbors outside 'vertices' are ignored

    Returns:
        Dictionary mapping each vertex to its assigned color (0-indexed)

    Example:
        >>> vertices = [0, 2, 5]
        >>> adjacency = {0: {2}, 2: {0, 5}, 5: {2}}
        >>> colors = dsatur_coloring(vertices, adjacency)
        >>> colors  # {0: 0, 2: 1, 5: 0}
    """
    if not vertices:
        return {}

    vertex_set = set(vertices)
    n = len(vertices)

    # Build induced adjacency (only edges within the vertex set)
    induced_adj: dict[int, set[int]] = {}
    for v in vertices:
        induced_adj[v] = adjacency.get(v, set()) & vertex_set

    # Track coloring state
    color: dict[int, int] = {}  # vertex -> color
    neighbor_colors: dict[int, set[int]] = {v: set() for v in vertices}  # saturation tracking

    # Compute initial degrees in induced subgraph (for tie-breaking)
    degree = {v: len(induced_adj[v]) for v in vertices}

    # Color all vertices
    for _ in range(n):
        # Select vertex with max saturation, break ties by max degree
        best_vertex: Optional[int] = None
        best_saturation = -1
        best_degree = -1

        for v in vertices:
            if v in color:
                continue  # already colored

            sat = len(neighbor_colors[v])
            deg = degree[v]

            if sat > best_saturation or (sat == best_saturation and deg > best_degree):
                best_vertex = v
                best_saturation = sat
                best_degree = deg

        if best_vertex is None:
            break  # all colored

        # Find smallest available color for best_vertex
        used_colors = neighbor_colors[best_vertex]
        c = 0
        while c in used_colors:
            c += 1

        # Assign color
        color[best_vertex] = c

        # Update saturation of uncolored neighbors
        for neighbor in induced_adj[best_vertex]:
            if neighbor not in color:
                neighbor_colors[neighbor].add(c)

    return color


def count_colors(coloring: dict[int, int]) -> int:
    """
    Count the number of distinct colors used in a coloring.

    Args:
        coloring: Dictionary mapping vertices to colors

    Returns:
        Number of distinct colors (0 if empty)
    """
    if not coloring:
        return 0
    return len(set(coloring.values()))


def verify_coloring(
    vertices: list[int],
    adjacency: dict[int, set[int]],
    coloring: dict[int, int],
) -> bool:
    """
    Verify that a coloring is valid (no adjacent vertices share a color).

    Args:
        vertices: List of vertex IDs
        adjacency: Adjacency dict
        coloring: Color assignment to verify

    Returns:
        True if coloring is valid, False otherwise
    """
    vertex_set = set(vertices)

    for v in vertices:
        if v not in coloring:
            return False  # vertex not colored

        v_color = coloring[v]
        for neighbor in adjacency.get(v, set()):
            if neighbor in vertex_set and neighbor in coloring:
                if coloring[neighbor] == v_color:
                    return False  # conflict

    return True
