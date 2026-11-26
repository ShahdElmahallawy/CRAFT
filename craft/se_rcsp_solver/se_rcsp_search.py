"""SE-RCSP solver — the paper's Algorithm 2.

This file is fixed. Everything query-specific lives in the four generated
functions on the Algorithm object: Start, Expand, IsGoal, Dominates.

Paper -> code:  OPEN -> OPEN,  F -> frontier,  s' -> s_prime,  s_e -> s_e,
                f -> .objective,  f-hat -> .bound,  pi -> .prev.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

NO_BOUND = float("-inf")        # default f-hat: never prunes


@dataclass
class SearchResult:
    path: Optional[List[Any]]           # the answer: nodes, source to destination
    path_states: Optional[List[Any]]    # the same chain, as states
    states: List[Any] = field(default_factory=list)   # every state built; index = id


def reconstruct_path(states: List[Any], state_id: Optional[int]):
    """ReconstructPath: follow pi back to the source."""
    if state_id is None:
        return None, None
    ids: List[int] = []
    while state_id is not None:
        ids.append(state_id)
        state_id = states[state_id].prev
    ids.reverse()
    chain = [states[i] for i in ids]
    return [s.node for s in chain], chain


def se_rcsp_search(algorithm: Any, *, monotonic: bool = True) -> SearchResult:
    """Find a cost-optimal feasible path, or an empty result if none exists.

    monotonic=True (default): f only grows (distance, time), so the first goal
        popped is optimal — return it.
    monotonic=False: f only shrinks (a maximized objective stored negated), so a
        popped goal proves nothing. Keep searching and return the best complete
        route found.

    Set monotonic from the query's objective; it can't be read off the start
    state, since both cases start at f = 0.
    """
    states: List[Any] = []                  # index = state id
    frontier: Dict[Any, List[int]] = {}     # F: node -> live state ids
    OPEN: List[Tuple[float, int]] = []      # min-heap of (f, state id)
    retired: set = set()                    # dominated while queued; skip on pop

    s_0 = algorithm.start()                                     # line 1
    states.append(s_0.with_prev(None))
    frontier[s_0.node] = [0]                                    # line 2
    heapq.heappush(OPEN, (s_0.objective, 0))

    best_id: Optional[int] = None           # the state we return (best)
    # The loop records goals as they are generated, and start never gets there.
    if not monotonic and algorithm.is_goal(states[0]):
        best_id = 0

    while OPEN:                                                 # line 3
        _, s_id = heapq.heappop(OPEN)                           # line 4
        if s_id in retired:
            continue
        s = states[s_id]

        if monotonic and algorithm.is_goal(s):                  # lines 5-6
            # Smallest f in OPEN and f only grows — nothing can beat it.
            best_id = s_id                                      # line 7
            break
        # Non-monotonic: best is recorded as successors are generated, and a
        # goal is expanded like any other state, so a path may pass through v_d.

        for s_prime in algorithm.expand(s):                     # line 9
            if monotonic:
                s_prime_is_goal = False     # no incumbent to maintain
            else:
                s_prime_is_goal = algorithm.is_goal(s_prime)    # line 13
                if not s_prime_is_goal and best_id is not None:
                    # f-hat is the best f any route through s_prime could still
                    # reach. We minimize, so if even that can't beat best, drop
                    # s_prime. Left at -inf it never prunes.
                    if s_prime.bound >= states[best_id].objective:
                        continue                                # lines 16-17

            # Drop s_prime if some s_e at this node dominates it; retire the
            # ones it dominates. Incomparable states are both kept.
            node = s_prime.node
            dominated = False
            survivors: List[int] = []
            for s_e_id in frontier.get(node, ()):               # line 18
                s_e = states[s_e_id]
                if algorithm.dominates(s_e, s_prime):
                    dominated = True
                    break
                if algorithm.dominates(s_prime, s_e):
                    retired.add(s_e_id)
                else:
                    survivors.append(s_e_id)
            if dominated:
                continue                                        # line 19

            new_id = len(states)
            states.append(s_prime.with_prev(s_id))              # line 12 (pi)
            frontier[node] = survivors + [new_id]               # line 20

            # Paper records this before the dominance check; we wait until
            # s_prime has an id. A dominated goal isn't recorded, but its
            # dominator is no worse, so best is already at least as good.
            if s_prime_is_goal and (                            # lines 14-15
                best_id is None
                or s_prime.objective < states[best_id].objective
            ):
                best_id = new_id

            heapq.heappush(OPEN, (s_prime.objective, new_id))   # line 21

    path, path_states = reconstruct_path(states, best_id)       # lines 7, 22
    return SearchResult(path=path, path_states=path_states, states=states)
