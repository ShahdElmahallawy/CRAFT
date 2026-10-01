# Problem Description
You are a code generator. Your job is to fill four specific bodies inside a fixed template that solves constrained routing problems on a road graph: the body of `State.start`, and the bodies of three `Algorithm` methods — `expand`, `dominates`, and `is_goal`. The rest of the template, the search core, and the State class interface are fixed and must not be referenced or redefined.

## Input
You receive the query's `input.json`. Its fields define the objective and constraints and are authoritative. The fields vary from problem to problem, so always read the field names from `input.json` and use them exactly.

Three fields are always present:
- `src`: start node id
- `dst`: destination node id
- `objective`: optimization objective (`minimize_travel_time`, `minimize_distance`, or `maximize_poi`)

## Goal
- Start at `src`, finish at `dst`.
- Satisfy **all** constraints present in the input — every constraint is a hard requirement your code must actively track and check; if a limit is given in the input, no valid move may exceed it
- Minimize total travel time (or distance, depending on the problem)
- For `maximize_poi`, collect as many POIs as fit the time budget. Category caps are upper limits: a route filling some or none of them is valid.

## Unit conventions
Already applied — do not convert:
- All distance values in `input.json` are in meters.
- All time values in `input.json` are in seconds.

## Definitions
### Objective
- The quantity you actually minimize, or maximize.
- Examples: total travel time, total distance, number of POIs.
- This is what you put in Dijkstra's priority queue key.
- For `maximize_poi`, set `objective = -count`, where count is the number of POIs collected so far (the core uses a min-heap).

### Constraints
- Something limited or constrained, with bounds or feasibility conditions.
- Examples: remaining battery, POI progress index OR any additional specifications the user may have.
- Used in dominance tests and feasibility checks.
- Do not store or compare specific node ids (or the route/path) in constraints — they explode the search. Exception: for `maximize_poi`, keep a set of collected POI node ids to prevent collecting the same POI twice. Use it only in `expand`, never in `dominates`.

## State
State class is fixed in the template with:
- `state.node`
- `state.prev`
- `state.constraints` (dictionary for resource/progress variables)
- `@property objective`
- `update_prev(prev_id)`
- `State.start(origin, init_context)`

### What the LLM must decide
- Initialization of objective inside `State.start(...)`.
- Initial contents of `state.constraints` inside `State.start(...)`.
- How constraints are updated/checked in `expand`, and used in `dominates`/`is_goal`.

## Roles of the three core functions
- `is_goal(state)`: returns `True` only when this state has reached the destination and satisfies all user-required constraints. For `maximize_poi`, `is_goal` is only an early-stop signal: return True when the state is at the destination and has filled every category cap, since no route can collect more. This is not the definition of a valid solution — any route reaching the destination within budget is valid, even if caps are unfilled.
- `expand(state)`: generates only the next valid states by moving to neighbors, checking whether each move is allowed by the constraints, and updating objective/constraints. At a node that qualifies as a POI stop, yield both successors — one that stops and collects, one that drives past — since a later instance of the same type may lead to a better route. 
- `dominates(a, b)`: returns True when state a is no worse than b on the objective and on every tracked constraint resource, and strictly better on at least one, so b is pruned. "No worse" means a can still do everything b can from this node with at least as good a final result: less of a consumed resource (time, distance, battery used) is better, and more progress (stops completed, items collected per category) is better.

**Bound pruning (required for `maximize_poi` — the search is far too slow without it).**
In `expand`, apply this to **every** successor before yielding it — including the one that drives past a node without stopping. Routes reach the destination by driving into it, and that is where the incumbent is recorded, so skipping drive-past successors disables pruning.
For each successor, construct the `State` fully (node/constraints/objective), then:
1. `rem_time = max(0, time_budget − successor_elapsed_time − min_time_to_dest(self.G, successor_node, self.destination))`
2. `slots` = total remaining room under all category caps. If `slots == 0`, set `opt_rem = 0` and skip step 3.
3. `min_dwell` = the smallest `visit_time` among categories still **under** their cap. `opt_rem = min(slots, rem_time // min_dwell)`; if `min_dwell == 0`, `opt_rem = slots`.
4. `optimistic_objective = -(successor_count + opt_rem)`
Then call `self.bound_prune(successor_state, optimistic_objective)`; if it returns `True`, do not yield that successor.
The incumbent is maintained inside `bound_prune` — never manage it yourself, and never read or write `self._best`. Your estimate must stay a true upper bound on what is still collectible, or the prune can discard the optimum.

### Function return contracts
- `State.start(origin, init_context)`:
- must return exactly one `State` object: `State(origin, None, constraints, objective)`
- `dominates(a, b)`: must return only a boolean
- `is_goal(state)`: must return only a boolean
- `expand(state)`:
- must yield only `State` objects
- each yielded state uses `prev=None` (the core assigns parent ids)
- each yielded state must provide updated `constraints` and updated `objective`

## General rules
1. Charging: passing through a charger node (detected via `self.type_nodes.get('charging_station', set())`) resets battery to full capacity.
2. POI visit penalty/time: `visit_time` is a **per-category mapping in seconds** (e.g. `{'pharmacy': 900, 'library': 3600}`). When you collect a POI of category `c`, add `self.visit_time[c]` seconds to the elapsed time — different categories cost different amounts. (If `visit_time` is a single number instead of a mapping, use it for every stop; if it is absent, use 1800 seconds.) Do not invent other values.
3. Invalid edges: skip NaN, missing, or non-positive costs.
4. Enforce each limit against the amount accumulated along the route so far, not a single step; reset a per-leg accumulator at each stop it applies to.

## Available self attributes
- `self.G`: road graph (`networkx.MultiDiGraph`)
- `self.destination`: destination node id
- `self.weight_key`: edge weight key (`"travel_time"` or `"length"`)
- Note: this is derived internally from `objective` (legacy `weight_key` is still accepted if present).
- `self.default_speed_kph`: fallback speed for edges missing travel_time
- Charging stations are POIs with category `charging_station`.
- Every field in `input.json` is also available as `self.{field_name}`.
- `self.possible_targets`: `Dict[int, Set]` — maps each stop index to the set of valid candidate node IDs for that stop. This is the way to check POI eligibility (always present whenever `poi_types` is in the input). It already incorporates type filtering, structured attribute filters, and description-based filters. Use `neighbor in self.possible_targets.get(i, set())` to check whether a neighbor qualifies as stop `i`. **For ordered templates** (the query asks to visit types in a specific sequence), advance a single sequential stop index — visit stop 0, then 1, then 2. **For unordered templates** (input `ordered: false`, or the query says "in any order"), check every not-yet-visited stop index and accept any match — do NOT force a sequence. Stop index `i` has category `self.poi_types[i]`; get a node's category from the index it matched, not from `pois_gdf`.
- `self.pois_gdf`: GeoDataFrame of all POIs with their attributes (one row per POI). The schema (provided in the query context) describes which columns exist and their value formats. The column linking each POI row to a graph node is `nearest_node_osmid`. Column values may need normalization or type conversion before use. Missing values come back as `NaN`. Build any per-column lookups once (not inside every expand call). **CRITICAL**: Do NOT use or invent fields that do not exist in the schema.

Always read the field names from `input.json` and use them exactly. Do not assume fixed names and do not use aliases not present in `input.json`.

## Available helpers
First three functions you can call them directly, not as self methods
- `edge_weight(data, self.weight_key, self.default_speed_kph)` → objective edge cost based on `self.weight_key`
- `edge_weight(data, "length", self.default_speed_kph)` → distance (meters)
- `min_time_to_dest(self.G, node, self.destination)` → lower-bound travel time in **seconds** from `node` to the destination; never over-estimates, so it's safe inside an optimistic pruning bound.
- `self.bound_prune(successor_state, optimistic_objective)` → returns True if this successor cannot beat the best complete route found so far (skip it), False otherwise. Pass the State object, not a node id. It maintains the incumbent internally; call it on every successor before yielding.

## Critical patterns
### Edge iteration — use exactly this form
```python
for _, neighbor, _, data in self.G.out_edges(state.node, keys=True, data=True):
```

## Restrictions
- Do not add imports, helper functions, or class definitions.
- Do not reference files, environment variables, or network calls.
- Use only self attributes and provided helpers.