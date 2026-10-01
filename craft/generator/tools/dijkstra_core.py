from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


@dataclass
class SearchResult:
    states: List[Any]
    goal_state_id: Optional[int]

    @property
    def goal_node_id(self) -> Optional[Any]:
        if self.goal_state_id is None:
            return None
        return self.states[self.goal_state_id].node


def reconstruct_path(
    states: List[Any],
    goal_state_id: Optional[int],
    *,
    node_getter: Callable[[Any], Any],
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    if goal_state_id is None:
        return None, None

    path_ids: List[int] = []
    cursor: Optional[int] = goal_state_id
    while cursor is not None:
        path_ids.append(cursor)
        cursor = states[cursor].prev
    path_ids = list(reversed(path_ids))

    path_states = [states[i] for i in path_ids]
    path_nodes = [node_getter(s) for s in path_states]
    return path_nodes, path_states


def dijkstra_search(
    *,
    start_state: Any,
    expand: Callable[[Any], Iterable[Any]],
    dominates: Callable[[Any, Any], bool],
    is_goal: Callable[[Any], bool],
) -> SearchResult:
    states: List[Any] = []
    nodes_id: Dict[object, List[int]] = {}
    heap: List[Tuple[float, int]] = []

    start_state = start_state.update_prev(None)
    state_id = len(states)
    states.append(start_state)
    key = start_state.node
    nodes_id.setdefault(key, []).append(state_id)
    heapq.heappush(heap, (start_state.objective, state_id))

    goal_state_id: Optional[int] = None
    dead=set()

    while heap:
        _, state_id = heapq.heappop(heap)
        if state_id in dead:        # was dominated after being queued
            continue
        current = states[state_id]

        if is_goal(current):
            goal_state_id = state_id
            break

        for next_state in expand(current):
            key = next_state.node
            existing_state_ids = nodes_id.get(key, [])
            dominated = False
            kept_state_ids: List[int] = []
            for ex_id in existing_state_ids:
                ex_state = states[ex_id]
                if dominates(ex_state, next_state):
                    dominated = True
                    break
                if dominates(next_state, ex_state):
                    dead.add(ex_id)             # old state is worse: skip it later
                else:
                    kept_state_ids.append(ex_id)
            if dominated:
                continue
            nodes_id[key] = kept_state_ids

            new_state_id = len(states)
            states.append(next_state.update_prev(state_id))
            nodes_id[key].append(new_state_id)
            heapq.heappush(heap, (next_state.objective, new_state_id))

    return SearchResult(states=states, goal_state_id=goal_state_id)
