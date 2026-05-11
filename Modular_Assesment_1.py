"""
Intelligent Campus Assistant - Path Optimisation System
IHC Modular Assignment No. 1 | Topic: Search and Reasoning
"""

import heapq
from collections import deque
import math

# ─────────────────────────────────────────────
#  1. CAMPUS GRAPH DEFINITION
# ─────────────────────────────────────────────

# Nodes: Campus locations
LOCATIONS = {
    "Main Gate":        (0,  0),
    "Library":          (2,  3),
    "Cafeteria":        (4,  1),
    "Admin Block":      (2,  6),
    "Science Block":    (5,  5),
    "Sports Complex":   (7,  2),
    "Hostel A":         (0,  6),
    "Hostel B":         (1,  8),
    "Auditorium":       (4,  8),
    "Parking Lot":      (7,  0),
}

# Edges: (location_a, location_b, base_distance)
EDGES = [
    ("Main Gate",     "Library",        3.6),
    ("Main Gate",     "Cafeteria",      4.1),
    ("Main Gate",     "Parking Lot",    7.0),
    ("Library",       "Admin Block",    3.0),
    ("Library",       "Cafeteria",      2.8),
    ("Library",       "Science Block",  3.6),
    ("Cafeteria",     "Sports Complex", 3.2),
    ("Cafeteria",     "Science Block",  4.1),
    ("Admin Block",   "Hostel A",       2.2),
    ("Admin Block",   "Auditorium",     2.8),
    ("Admin Block",   "Science Block",  3.2),
    ("Science Block", "Auditorium",     3.6),
    ("Science Block", "Sports Complex", 2.8),
    ("Hostel A",      "Hostel B",       2.1),
    ("Hostel B",      "Auditorium",     3.2),
    ("Sports Complex","Parking Lot",    2.2),
]

# Build adjacency list
def build_graph(edges, weights=None):
    """Build adjacency list graph. weights override base distances."""
    graph = {loc: {} for loc in LOCATIONS}
    for a, b, base_dist in edges:
        cost = weights.get((a, b), weights.get((b, a), base_dist)) if weights else base_dist
        graph[a][b] = cost
        graph[b][a] = cost
    return graph


# ─────────────────────────────────────────────
#  2. REASONING ENGINE — DYNAMIC WEIGHT ADJUSTER
# ─────────────────────────────────────────────

class ReasoningEngine:
    """
    Applies real-world factors to edge weights:
    - Crowd density  : multiplies weight on crowded paths
    - Weather        : prefers covered/indoor routes when raining
    - Events         : blocks or penalises paths near active events
    """

    # Covered/indoor path segments (safe from rain)
    COVERED_PATHS = {
        ("Library", "Admin Block"),
        ("Admin Block", "Hostel A"),
        ("Admin Block", "Auditorium"),
        ("Science Block", "Auditorium"),
    }

    # Paths near event-prone venues
    EVENT_ADJACENT = {
        "Auditorium": [("Science Block", "Auditorium"),
                       ("Hostel B", "Auditorium"),
                       ("Admin Block", "Auditorium")],
        "Sports Complex": [("Cafeteria", "Sports Complex"),
                           ("Science Block", "Sports Complex"),
                           ("Sports Complex", "Parking Lot")],
    }

    def __init__(self):
        self.crowd_levels   = {loc: 0.0 for loc in LOCATIONS}  # 0–1
        self.is_raining     = False
        self.active_events  = set()   # venue names

    def set_conditions(self, crowd_levels=None, is_raining=False, active_events=None):
        if crowd_levels:
            self.crowd_levels.update(crowd_levels)
        self.is_raining    = is_raining
        self.active_events = set(active_events) if active_events else set()

    def compute_weights(self):
        """Return edge-weight dictionary reflecting current conditions."""
        weights = {}

        for a, b, base in EDGES:
            cost = base

            # --- Crowd penalty ---
            avg_crowd = (self.crowd_levels.get(a, 0) + self.crowd_levels.get(b, 0)) / 2
            if avg_crowd > 0.7:
                cost *= 1.8   # heavy crowd → nearly double traversal time
            elif avg_crowd > 0.4:
                cost *= 1.3   # moderate crowd

            # --- Rain penalty (prefer covered paths) ---
            if self.is_raining:
                edge = (a, b)
                rev  = (b, a)
                if edge not in self.COVERED_PATHS and rev not in self.COVERED_PATHS:
                    cost *= 1.5   # outdoor path less desirable in rain

            # --- Event penalty ---
            for venue, adjacent in self.EVENT_ADJACENT.items():
                if venue in self.active_events:
                    if (a, b) in adjacent or (b, a) in adjacent:
                        cost *= 2.0   # congestion around event venues

            weights[(a, b)] = cost
            weights[(b, a)] = cost

        return weights

    def describe_conditions(self):
        lines = []
        high_crowd = [loc for loc, v in self.crowd_levels.items() if v > 0.5]
        if high_crowd:
            lines.append(f"  🚶 High crowd at: {', '.join(high_crowd)}")
        if self.is_raining:
            lines.append("  🌧  Weather: Raining (outdoor paths penalised)")
        if self.active_events:
            lines.append(f"  🎭 Active events at: {', '.join(self.active_events)}")
        if not lines:
            lines.append("  ☀️  Normal conditions — no special factors")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  3. BFS — SHORTEST HOPS (UNWEIGHTED)
# ─────────────────────────────────────────────

def bfs(graph, start, goal):
    """Returns (path, hop_count) using BFS (fewest edges)."""
    if start == goal:
        return [start], 0

    visited = {start}
    queue   = deque([(start, [start])])

    while queue:
        node, path = queue.popleft()
        for neighbour in graph[node]:
            if neighbour not in visited:
                new_path = path + [neighbour]
                if neighbour == goal:
                    return new_path, len(new_path) - 1
                visited.add(neighbour)
                queue.append((neighbour, new_path))

    return None, float('inf')   # no path found


# ─────────────────────────────────────────────
#  4. A* — OPTIMAL WEIGHTED PATH
# ─────────────────────────────────────────────

def euclidean_heuristic(node, goal):
    """Straight-line distance between node and goal (admissible heuristic)."""
    x1, y1 = LOCATIONS[node]
    x2, y2 = LOCATIONS[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def astar(graph, start, goal):
    """Returns (path, total_cost) using A* with Euclidean heuristic."""
    if start == goal:
        return [start], 0.0

    # Priority queue: (f_score, g_score, node, path)
    open_set = [(euclidean_heuristic(start, goal), 0.0, start, [start])]
    visited  = {}   # node → best g_score seen

    while open_set:
        f, g, node, path = heapq.heappop(open_set)

        if node == goal:
            return path, round(g, 2)

        if node in visited and visited[node] <= g:
            continue
        visited[node] = g

        for neighbour, edge_cost in graph[node].items():
            new_g = g + edge_cost
            if neighbour not in visited or visited[neighbour] > new_g:
                h     = euclidean_heuristic(neighbour, goal)
                new_f = new_g + h
                heapq.heappush(open_set, (new_f, new_g, neighbour, path + [neighbour]))

    return None, float('inf')


# ─────────────────────────────────────────────
#  5. SIMULATION RUNNER
# ─────────────────────────────────────────────

engine = ReasoningEngine()

def run_simulation(title, start, goal, crowd_levels=None,
                   is_raining=False, active_events=None):
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {title}")
    print(f"  Route   : {start}  →  {goal}")
    print(f"{'='*60}")

    # Apply conditions
    engine.set_conditions(
        crowd_levels  = crowd_levels  or {},
        is_raining    = is_raining,
        active_events = active_events or []
    )
    print("\nConditions:")
    print(engine.describe_conditions())

    # Build weighted graph
    weights = engine.compute_weights()
    graph   = build_graph(EDGES, weights)

    # BFS (unweighted hops)
    bfs_path, bfs_hops = bfs(build_graph(EDGES), start, goal)   # always unweighted

    # A* (weighted, condition-aware)
    astar_path, astar_cost = astar(graph, start, goal)

    print(f"\n📍 BFS Path  ({bfs_hops} hops):")
    print(f"   {' → '.join(bfs_path)}")

    print(f"\n⭐ A* Path   (cost = {astar_cost:.2f}):")
    print(f"   {' → '.join(astar_path)}")

    if bfs_path != astar_path:
        print("\n  💡 Reasoning: A* chose a different route than BFS because")
        print("     real-world conditions increased the cost of the direct path.")
    else:
        print("\n  ✅ Both algorithms agree on the optimal route.")

    return {
        "scenario"  : title,
        "start"     : start,
        "goal"      : goal,
        "bfs_path"  : bfs_path,
        "bfs_hops"  : bfs_hops,
        "astar_path": astar_path,
        "astar_cost": astar_cost,
    }


# ─────────────────────────────────────────────
#  6. SCENARIOS
# ─────────────────────────────────────────────

results = []

results.append(run_simulation(
    title        = "Scenario 1 — Normal Day (No Special Conditions)",
    start        = "Main Gate",
    goal         = "Hostel B",
))

results.append(run_simulation(
    title        = "Scenario 2 — Rainy Weather",
    start        = "Main Gate",
    goal         = "Science Block",
    is_raining   = True,
))

results.append(run_simulation(
    title        = "Scenario 3 — Auditorium Event (Evening Concert)",
    start        = "Hostel A",
    goal         = "Cafeteria",
    active_events= ["Auditorium"],
    crowd_levels = {"Auditorium": 0.9, "Admin Block": 0.7},
))

results.append(run_simulation(
    title        = "Scenario 4 — Peak Hours (Multiple Crowded Zones)",
    start        = "Parking Lot",
    goal         = "Library",
    crowd_levels = {
        "Cafeteria"     : 0.85,
        "Main Gate"     : 0.80,
        "Science Block" : 0.75,
    },
))

print(f"\n{'='*60}")
print("  SUMMARY TABLE")
print(f"{'='*60}")
print(f"{'Scenario':<45} {'BFS Hops':>8} {'A* Cost':>8}")
print("-" * 63)
for r in results:
    short = r['scenario'].split('—')[1].strip()
    print(f"{short:<45} {r['bfs_hops']:>8} {r['astar_cost']:>8.2f}")
