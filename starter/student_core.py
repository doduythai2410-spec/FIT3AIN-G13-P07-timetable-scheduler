"""Graded AI core for Project 07 (University Timetable Scheduler).

Formalizes university course timetabling as a Constraint Satisfaction Problem
(CSP) <X, D, C> and implements four systematic, offline search strategies from
first principles (no external CSP libraries or LLM APIs):

- ``naive``: Baseline chronological backtracking in fixed CSV course order.
- ``midterm_heuristic``: Midterm MRV + copy-on-branch Forward Checking
  (retained for controlled Midterm-vs-Final experimental evaluation).
- ``heuristic``: Final AI core combining Fast AC-3 preprocessing, MRV with
  Dynamic Degree tie-breaking, and incremental neighbor-indexed Forward Checking
  with trail-based rollback.
- ``optimized``: Final AI core (``heuristic``) followed by deterministic,
  hard-constraint-preserving steepest-ascent local search with variance-smoothed
  plateau breaking to maximize ``soft_constraint_score``.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

try:
    import loaders
except ModuleNotFoundError:
    from starter import loaders

Choice = Tuple[str, str, str]  # (lecturer_id, room_id, slot_id)

SEARCH_STATS: Dict[str, object] = {
    "nodes": 0,
    "attempts": 0,
    "backtracks": 0,
    "pruned_values": 0,
    "ac3_pruned": 0,
    "dead_ends": 0,
    "max_depth": 0,
    "runtime_ms": 0.0,
    "search_ms": 0.0,
    "opt_ms": 0.0,
    "opt_steps": 0,
    "initial_soft_score": None,
    "final_soft_score": None,
    "strategy": "naive",
    "status": "NOT_RUN",
}


def is_consistent(
    instance: dict,
    courses_by_id: dict,
    course_id: str,
    choice: Choice,
    assignment: Dict[str, Choice],
) -> bool:
    """Test a locally valid candidate against every *assigned* course.

    Choices must come from ``loaders.candidate_domain``, which enforces
    lecturer eligibility, room type/capacity, and availability windows.
    """
    lecturer_id, room_id, slot_id = choice
    cohort = courses_by_id[course_id]["cohort"]
    for other_id, (other_lecturer, other_room, other_slot) in assignment.items():
        if other_slot == slot_id:
            if (
                other_lecturer == lecturer_id
                or other_room == room_id
                or courses_by_id[other_id]["cohort"] == cohort
            ):
                return False

    for rule in instance["constraints"].get("extra_constraints", []):
        if rule.get("type") != "lecturer_daily_limit":
            raise ValueError(f"Unknown extra hard constraint: {rule.get('type')}")
        if lecturer_id != rule["lecturer_id"]:
            continue
        day = instance["slot_info"][slot_id]["day"]
        previous = sum(
            other_lecturer == lecturer_id
            and instance["slot_info"][other_slot]["day"] == day
            for other_lecturer, _, other_slot in assignment.values()
        )
        if previous >= int(rule.get("max_per_day", 1)):
            return False
    return True


def _extract_daily_limits(instance: dict) -> Dict[str, int]:
    """Validate and index all ``lecturer_daily_limit`` extra hard constraints."""
    limits: Dict[str, int] = {}
    for rule in instance["constraints"].get("extra_constraints", []):
        if rule.get("type") != "lecturer_daily_limit":
            raise ValueError(f"Unknown extra hard constraint: {rule.get('type')}")
        lid = rule["lecturer_id"]
        max_per_day = int(rule.get("max_per_day", 1))
        limits[lid] = min(limits[lid], max_per_day) if lid in limits else max_per_day
    return limits


def _build_conflict_graph(
    course_order: List[str],
    courses: Dict[str, dict],
    domains: Dict[str, List[Choice]],
) -> Dict[str, Set[str]]:
    """Build adjacency list of courses that can share a lecturer, room, or cohort."""
    lecturers_of = {cid: {v[0] for v in domains[cid]} for cid in course_order}
    rooms_of = {cid: {v[1] for v in domains[cid]} for cid in course_order}
    neighbors: Dict[str, Set[str]] = {cid: set() for cid in course_order}

    for i, c1 in enumerate(course_order):
        coh1 = courses[c1]["cohort"]
        l1 = lecturers_of[c1]
        r1 = rooms_of[c1]
        for c2 in course_order[i + 1 :]:
            if (
                coh1 == courses[c2]["cohort"]
                or not l1.isdisjoint(lecturers_of[c2])
                or not r1.isdisjoint(rooms_of[c2])
            ):
                neighbors[c1].add(c2)
                neighbors[c2].add(c1)
    return neighbors


def _fast_ac3_preprocess(
    course_order: List[str],
    courses: Dict[str, dict],
    domains: Dict[str, List[Choice]],
    neighbors: Dict[str, Set[str]],
    slot_day: Dict[str, str],
    daily_limits: Dict[str, int],
    stats: Dict[str, object],
) -> bool:
    """Sound O(e * d) generalized arc-consistency (AC-3) preprocessing.

    Whenever all remaining candidates in domain D(Xj) force a common
    (lecturer, slot), (room, slot), slot (within the same cohort), or
    (lecturer, day) with max_per_day <= 1, any conflicting candidate in a
    neighboring domain D(Xi) has zero support in D(Xj) and is pruned prior
    to backtracking search.
    """
    queue: deque[str] = deque(course_order)
    in_queue: Set[str] = set(course_order)

    while queue:
        xj = queue.popleft()
        in_queue.discard(xj)
        dom_j = domains[xj]
        if not dom_j:
            return False

        first_l, first_r, first_s = dom_j[0]
        first_d = slot_day[first_s]

        forced_s = True
        forced_ls = True
        forced_rs = True
        forced_ld = first_l in daily_limits and daily_limits[first_l] <= 1

        for l_val, r_val, s_val in dom_j[1:]:
            if forced_s and s_val != first_s:
                forced_s = False
                forced_ls = False
                forced_rs = False
            else:
                if forced_ls and l_val != first_l:
                    forced_ls = False
                if forced_rs and r_val != first_r:
                    forced_rs = False
            if forced_ld and (l_val != first_l or slot_day[s_val] != first_d):
                forced_ld = False
            if not (forced_s or forced_ls or forced_rs or forced_ld):
                break

        if not (forced_s or forced_ls or forced_rs or forced_ld):
            continue

        coh_j = courses[xj]["cohort"]
        for xi in neighbors[xj]:
            dom_i = domains[xi]
            if not dom_i:
                return False
            same_cohort = forced_s and (courses[xi]["cohort"] == coh_j)
            new_dom_i: List[Choice] = []
            for v in dom_i:
                li, ri, si = v
                if same_cohort and si == first_s:
                    continue
                if forced_ls and li == first_l and si == first_s:
                    continue
                if forced_rs and ri == first_r and si == first_s:
                    continue
                if forced_ld and li == first_l and slot_day[si] == first_d:
                    continue
                new_dom_i.append(v)

            pruned = len(dom_i) - len(new_dom_i)
            if pruned > 0:
                domains[xi] = new_dom_i
                stats["ac3_pruned"] = int(stats["ac3_pruned"]) + pruned
                stats["pruned_values"] = int(stats["pruned_values"]) + pruned
                if not new_dom_i:
                    stats["dead_ends"] = int(stats["dead_ends"]) + 1
                    return False
                if xi not in in_queue:
                    queue.append(xi)
                    in_queue.add(xi)
    return True


def optimize_timetable(
    instance: dict,
    initial_assignment: Dict[str, Choice],
    domains: Optional[Dict[str, List[Choice]]] = None,
    max_iterations: int = 100,
) -> Tuple[Dict[str, Choice], int]:
    """Deterministic steepest-ascent local search over valid timetables.

    Maintains 100% hard-constraint feasibility at every move using O(1)
    incremental conflict tables and incremental objective state. Uses a
    lexicographic objective ``(primary_soft_score, -sum_sq_lecturer_loads)``
    so hill climbing smoothly escapes flat ``max_load - min_load`` plateaus.
    """
    configured = {
        rule["name"] if isinstance(rule, dict) else rule
        for rule in instance["constraints"].get("soft_constraints", [])
    }
    if not configured or not initial_assignment:
        return dict(initial_assignment), 0
    unknown = configured - {"cohort_daily_spread", "lecturer_load_balance"}
    if unknown:
        raise ValueError(f"Unsupported soft constraint(s): {sorted(unknown)}")

    use_spread = "cohort_daily_spread" in configured
    use_balance = "lecturer_load_balance" in configured
    n_metrics = int(use_spread) + int(use_balance)

    courses = {c["course_id"]: c for c in instance["courses"]}
    course_order = [c["course_id"] for c in instance["courses"]]
    slot_day = {sid: info["day"] for sid, info in instance["slot_info"].items()}
    daily_limits = _extract_daily_limits(instance)
    if domains is None:
        domains = {
            cid: sorted(loaders.candidate_domain(courses[cid], instance))
            for cid in course_order
        }

    cohort_sizes: Dict[str, int] = Counter(c["cohort"] for c in instance["courses"])
    num_cohorts = len(cohort_sizes)
    all_lecturers = list(instance["lecturers"].keys())
    total_courses = max(1, len(course_order))

    current = dict(initial_assignment)
    used_ls: Set[Tuple[str, str]] = set()
    used_rs: Set[Tuple[str, str]] = set()
    used_cs: Set[Tuple[str, str]] = set()
    lect_day_count: Dict[Tuple[str, str], int] = defaultdict(int)

    # Incremental soft-objective state
    cohort_day_count: Dict[Tuple[str, str], int] = defaultdict(int)
    cohort_distinct_days: Dict[str, int] = defaultdict(int)
    lect_load: Dict[str, int] = {lid: 0 for lid in all_lecturers}

    for cid, (l_id, r_id, s_id) in current.items():
        coh = courses[cid]["cohort"]
        day = slot_day[s_id]
        used_ls.add((l_id, s_id))
        used_rs.add((r_id, s_id))
        used_cs.add((coh, s_id))
        if l_id in daily_limits:
            lect_day_count[(l_id, day)] += 1
        if cohort_day_count[(coh, day)] == 0:
            cohort_distinct_days[coh] += 1
        cohort_day_count[(coh, day)] += 1
        lect_load[l_id] += 1

    spread_ratio_sum = sum(
        cohort_distinct_days[coh] / size for coh, size in cohort_sizes.items()
    )
    sum_sq_load = sum(v * v for v in lect_load.values())

    def current_objective(s_sum: float, sq_load: int) -> Tuple[float, int]:
        total = 0.0
        if use_spread:
            total += (s_sum / num_cohorts) if num_cohorts else 1.0
        if use_balance:
            vals = lect_load.values()
            bal = 1.0 - (max(vals) - min(vals)) / total_courses if all_lecturers else 1.0
            total += bal
        primary = round(100.0 * total / n_metrics, 6)
        # Secondary tie-breaker: minimize variance of lecturer loads
        return (primary, -sq_load)

    best_obj = current_objective(spread_ratio_sum, sum_sq_load)
    steps = 0

    for _ in range(max_iterations):
        best_move = None
        best_move_obj = best_obj

        # Neighborhood 1: single-course reassignment (O(|X| * |D|) with O(|L|) eval)
        for cid in course_order:
            old_l, old_r, old_s = current[cid]
            coh = courses[cid]["cohort"]
            coh_inv = 1.0 / cohort_sizes[coh]
            old_day = slot_day[old_s]

            # Remove old assignment from incremental state
            used_ls.remove((old_l, old_s))
            used_rs.remove((old_r, old_s))
            used_cs.remove((coh, old_s))
            if old_l in daily_limits:
                lect_day_count[(old_l, old_day)] -= 1

            cohort_day_count[(coh, old_day)] -= 1
            if cohort_day_count[(coh, old_day)] == 0:
                cohort_distinct_days[coh] -= 1
                spread_ratio_sum -= coh_inv

            old_load_val = lect_load[old_l]
            sum_sq_load -= 2 * old_load_val - 1
            lect_load[old_l] = old_load_val - 1

            for cand in domains[cid]:
                if cand == (old_l, old_r, old_s):
                    continue
                nl, nr, ns = cand
                if (nl, ns) in used_ls or (nr, ns) in used_rs or (coh, ns) in used_cs:
                    continue
                nday = slot_day[ns]
                if nl in daily_limits and lect_day_count[(nl, nday)] >= daily_limits[nl]:
                    continue

                # Speculative O(1) update
                added_day = cohort_day_count[(coh, nday)] == 0
                cand_spread = spread_ratio_sum + (coh_inv if added_day else 0.0)
                nl_prev = lect_load[nl]
                lect_load[nl] = nl_prev + 1
                cand_sq = sum_sq_load + 2 * nl_prev + 1

                obj = current_objective(cand_spread, cand_sq)
                lect_load[nl] = nl_prev

                if obj > best_move_obj:
                    best_move_obj = obj
                    best_move = ("single", cid, cand)

            # Restore old assignment
            lect_load[old_l] = old_load_val
            sum_sq_load += 2 * old_load_val - 1
            if cohort_day_count[(coh, old_day)] == 0:
                cohort_distinct_days[coh] += 1
                spread_ratio_sum += coh_inv
            cohort_day_count[(coh, old_day)] += 1
            if old_l in daily_limits:
                lect_day_count[(old_l, old_day)] += 1
            used_ls.add((old_l, old_s))
            used_rs.add((old_r, old_s))
            used_cs.add((coh, old_s))

        if best_move is not None:
            _, cid, new_val = best_move
            old_l, old_r, old_s = current[cid]
            coh = courses[cid]["cohort"]
            coh_inv = 1.0 / cohort_sizes[coh]
            old_day = slot_day[old_s]

            used_ls.remove((old_l, old_s))
            used_rs.remove((old_r, old_s))
            used_cs.remove((coh, old_s))
            if old_l in daily_limits:
                lect_day_count[(old_l, old_day)] -= 1
            cohort_day_count[(coh, old_day)] -= 1
            if cohort_day_count[(coh, old_day)] == 0:
                cohort_distinct_days[coh] -= 1
                spread_ratio_sum -= coh_inv
            old_lv = lect_load[old_l]
            sum_sq_load -= 2 * old_lv - 1
            lect_load[old_l] = old_lv - 1

            nl, nr, ns = new_val
            nday = slot_day[ns]
            current[cid] = new_val
            used_ls.add((nl, ns))
            used_rs.add((nr, ns))
            used_cs.add((coh, ns))
            if nl in daily_limits:
                lect_day_count[(nl, nday)] += 1
            if cohort_day_count[(coh, nday)] == 0:
                cohort_distinct_days[coh] += 1
                spread_ratio_sum += coh_inv
            cohort_day_count[(coh, nday)] += 1
            new_lv = lect_load[nl]
            sum_sq_load += 2 * new_lv + 1
            lect_load[nl] = new_lv + 1

            best_obj = best_move_obj
            steps += 1
            continue

        # Neighborhood 2: pairwise timeslot swap when single moves plateau
        domain_sets = {cid: set(domains[cid]) for cid in course_order}
        for i, c1 in enumerate(course_order):
            l1, r1, s1 = current[c1]
            coh1 = courses[c1]["cohort"]
            d1 = slot_day[s1]
            for c2 in course_order[i + 1 :]:
                l2, r2, s2 = current[c2]
                if s1 == s2 or slot_day[s1] == slot_day[s2]:
                    continue
                coh2 = courses[c2]["cohort"]
                cand1 = (l1, r1, s2)
                cand2 = (l2, r2, s1)
                if cand1 not in domain_sets[c1] or cand2 not in domain_sets[c2]:
                    continue

                d2 = slot_day[s2]
                used_ls.remove((l1, s1)); used_rs.remove((r1, s1)); used_cs.remove((coh1, s1))
                used_ls.remove((l2, s2)); used_rs.remove((r2, s2)); used_cs.remove((coh2, s2))
                if l1 in daily_limits:
                    lect_day_count[(l1, d1)] -= 1
                if l2 in daily_limits:
                    lect_day_count[(l2, d2)] -= 1

                feasible = (
                    (l1, s2) not in used_ls
                    and (r1, s2) not in used_rs
                    and (coh1, s2) not in used_cs
                    and (l2, s1) not in used_ls
                    and (r2, s1) not in used_rs
                    and (coh2, s1) not in used_cs
                )
                if feasible and l1 in daily_limits and lect_day_count[(l1, d2)] + 1 > daily_limits[l1]:
                    feasible = False
                if feasible and l2 in daily_limits:
                    extra = 1 if (l1 == l2 and d2 == d1) else 0
                    if lect_day_count[(l2, d1)] + 1 + extra > daily_limits[l2]:
                        feasible = False

                if feasible and coh1 != coh2:
                    # Compute exact delta on spread_ratio_sum (lecturer loads unchanged)
                    delta = 0.0
                    if cohort_day_count[(coh1, d1)] == 1:
                        delta -= 1.0 / cohort_sizes[coh1]
                    if cohort_day_count[(coh1, d2)] == 0:
                        delta += 1.0 / cohort_sizes[coh1]
                    if cohort_day_count[(coh2, d2)] == 1:
                        delta -= 1.0 / cohort_sizes[coh2]
                    if cohort_day_count[(coh2, d1)] == 0:
                        delta += 1.0 / cohort_sizes[coh2]
                    obj = current_objective(spread_ratio_sum + delta, sum_sq_load)
                    if obj > best_move_obj:
                        best_move_obj = obj
                        best_move = ("swap", c1, cand1, c2, cand2)

                used_ls.add((l1, s1)); used_rs.add((r1, s1)); used_cs.add((coh1, s1))
                used_ls.add((l2, s2)); used_rs.add((r2, s2)); used_cs.add((coh2, s2))
                if l1 in daily_limits:
                    lect_day_count[(l1, d1)] += 1
                if l2 in daily_limits:
                    lect_day_count[(l2, d2)] += 1

        if best_move is None:
            break

        _, c1, cand1, c2, cand2 = best_move
        l1, r1, s1 = current[c1]
        l2, r2, s2 = current[c2]
        coh1, coh2 = courses[c1]["cohort"], courses[c2]["cohort"]
        d1, d2 = slot_day[s1], slot_day[s2]

        used_ls.remove((l1, s1)); used_rs.remove((r1, s1)); used_cs.remove((coh1, s1))
        used_ls.remove((l2, s2)); used_rs.remove((r2, s2)); used_cs.remove((coh2, s2))
        if l1 in daily_limits:
            lect_day_count[(l1, d1)] -= 1
        if l2 in daily_limits:
            lect_day_count[(l2, d2)] -= 1

        for coh, old_d, new_d in ((coh1, d1, d2), (coh2, d2, d1)):
            coh_inv = 1.0 / cohort_sizes[coh]
            cohort_day_count[(coh, old_d)] -= 1
            if cohort_day_count[(coh, old_d)] == 0:
                cohort_distinct_days[coh] -= 1
                spread_ratio_sum -= coh_inv
            if cohort_day_count[(coh, new_d)] == 0:
                cohort_distinct_days[coh] += 1
                spread_ratio_sum += coh_inv
            cohort_day_count[(coh, new_d)] += 1

        current[c1] = cand1
        current[c2] = cand2
        used_ls.add((cand1[0], cand1[2])); used_rs.add((cand1[1], cand1[2])); used_cs.add((coh1, cand1[2]))
        used_ls.add((cand2[0], cand2[2])); used_rs.add((cand2[1], cand2[2])); used_cs.add((coh2, cand2[2]))
        if cand1[0] in daily_limits:
            lect_day_count[(cand1[0], d2)] += 1
        if cand2[0] in daily_limits:
            lect_day_count[(cand2[0], d1)] += 1

        best_obj = best_move_obj
        steps += 1

    return current, steps


def solve_csp(
    instance: dict,
    strategy: str = "naive",
    node_limit: Optional[int] = None,
) -> Optional[Dict[str, Choice]]:
    """Return a complete hard-constraint-valid timetable, else ``None``.

    Supported strategies:
    - ``naive``: Baseline chronological backtracking in fixed CSV order.
    - ``midterm_heuristic``: Midterm MRV + copy-on-branch Forward Checking.
    - ``heuristic``: Final AI core (Fast AC-3 + MRV/Dynamic Degree + Incremental FC).
    - ``optimized``: Final AI core + hard-constraint-preserving local search optimizer.

    Node definition: one recursive search invocation (including the empty root).
    ``node_limit`` defaults to ``None`` (complete search); if hit, ``status``
    is recorded as ``"LIMIT_EXCEEDED"`` instead of ``"UNSAT"``.
    """
    valid_strategies = {"naive", "midterm_heuristic", "heuristic", "optimized"}
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {sorted(valid_strategies)}")

    start = perf_counter()
    stats: Dict[str, object] = {
        "nodes": 0,
        "attempts": 0,
        "backtracks": 0,
        "pruned_values": 0,
        "ac3_pruned": 0,
        "dead_ends": 0,
        "max_depth": 0,
        "runtime_ms": 0.0,
        "search_ms": 0.0,
        "opt_ms": 0.0,
        "opt_steps": 0,
        "initial_soft_score": None,
        "final_soft_score": None,
        "strategy": strategy,
        "status": "SEARCHING",
    }
    SEARCH_STATS.clear()
    SEARCH_STATS.update(stats)

    course_order = [course["course_id"] for course in instance["courses"]]
    courses = {course["course_id"]: course for course in instance["courses"]}
    input_index = {cid: pos for pos, cid in enumerate(course_order)}
    domains = {
        cid: sorted(loaders.candidate_domain(courses[cid], instance))
        for cid in course_order
    }
    slot_day = {sid: info["day"] for sid, info in instance["slot_info"].items()}
    daily_limits = _extract_daily_limits(instance)
    assignment: Dict[str, Choice] = {}
    limit_hit = False

    def search_naive_or_midterm(live_domains: dict) -> Optional[Dict[str, Choice]]:
        nonlocal limit_hit
        if node_limit is not None and int(stats["nodes"]) >= node_limit:
            limit_hit = True
            return None
        stats["nodes"] = int(stats["nodes"]) + 1
        stats["max_depth"] = max(int(stats["max_depth"]), len(assignment))
        if len(assignment) == len(course_order):
            return assignment.copy()
        available = [cid for cid in course_order if cid not in assignment]
        cid = (
            available[0]
            if strategy == "naive"
            else min(available, key=lambda key: (len(live_domains[key]), input_index[key]))
        )

        for choice in live_domains[cid]:
            if limit_hit:
                return None
            stats["attempts"] = int(stats["attempts"]) + 1
            if not is_consistent(instance, courses, cid, choice, assignment):
                continue
            assignment[cid] = choice
            if strategy == "midterm_heuristic":
                filtered = {}
                future_possible = True
                for next_cid in available:
                    if next_cid == cid:
                        continue
                    valid = [
                        v
                        for v in live_domains[next_cid]
                        if is_consistent(instance, courses, next_cid, v, assignment)
                    ]
                    stats["pruned_values"] = int(stats["pruned_values"]) + (
                        len(live_domains[next_cid]) - len(valid)
                    )
                    if not valid:
                        future_possible = False
                        stats["dead_ends"] = int(stats["dead_ends"]) + 1
                        break
                    filtered[next_cid] = valid
                result = search_naive_or_midterm(filtered) if future_possible else None
            else:
                result = search_naive_or_midterm(live_domains)
            if result is not None:
                return result
            del assignment[cid]
            stats["backtracks"] = int(stats["backtracks"]) + 1
        return None

    def solve_final_incremental() -> Optional[Dict[str, Choice]]:
        nonlocal limit_hit
        work_domains: Dict[str, List[Choice]] = {cid: list(domains[cid]) for cid in course_order}
        neighbors = _build_conflict_graph(course_order, courses, work_domains)

        if not _fast_ac3_preprocess(
            course_order, courses, work_domains, neighbors, slot_day, daily_limits, stats
        ):
            stats["nodes"] = 1
            return None

        unassigned: Set[str] = set(course_order)
        lect_day_count: Dict[Tuple[str, str], int] = defaultdict(int)
        cohort_of = {cid: courses[cid]["cohort"] for cid in course_order}

        def backtrack() -> Optional[Dict[str, Choice]]:
            nonlocal limit_hit
            if node_limit is not None and int(stats["nodes"]) >= node_limit:
                limit_hit = True
                return None
            stats["nodes"] = int(stats["nodes"]) + 1
            stats["max_depth"] = max(int(stats["max_depth"]), len(assignment))
            if not unassigned:
                return assignment.copy()

            # MRV with Dynamic Degree tie-breaking (highest unassigned neighbor count)
            best_cid = None
            best_key = None
            for cid in unassigned:
                d_len = len(work_domains[cid])
                dyn_deg = sum(1 for nb in neighbors[cid] if nb in unassigned)
                key = (d_len, -dyn_deg, input_index[cid])
                if best_key is None or key < best_key:
                    best_key = key
                    best_cid = cid

            assert best_cid is not None
            cid = best_cid
            unassigned.remove(cid)
            coh = cohort_of[cid]
            choices = work_domains[cid]

            for choice in choices:
                if limit_hit:
                    unassigned.add(cid)
                    return None
                stats["attempts"] = int(stats["attempts"]) + 1
                l_id, r_id, s_id = choice
                day = slot_day[s_id]

                assignment[cid] = choice
                limit_reached = False
                if l_id in daily_limits:
                    lect_day_count[(l_id, day)] += 1
                    limit_reached = lect_day_count[(l_id, day)] >= daily_limits[l_id]

                saved_domains: List[Tuple[str, List[Choice]]] = []
                wipeout = False

                for nb in neighbors[cid]:
                    if nb not in unassigned:
                        continue
                    nb_dom = work_domains[nb]
                    same_coh = cohort_of[nb] == coh
                    kept: List[Choice] = []
                    for v in nb_dom:
                        vl, vr, vs = v
                        if vs == s_id and (same_coh or vl == l_id or vr == r_id):
                            continue
                        if limit_reached and vl == l_id and slot_day[vs] == day:
                            continue
                        kept.append(v)

                    pruned = len(nb_dom) - len(kept)
                    if pruned > 0:
                        stats["pruned_values"] = int(stats["pruned_values"]) + pruned
                        saved_domains.append((nb, nb_dom))
                        work_domains[nb] = kept
                        if not kept:
                            wipeout = True
                            stats["dead_ends"] = int(stats["dead_ends"]) + 1
                            break

                result = backtrack() if not wipeout else None
                if result is not None:
                    return result

                for nb, old_dom in saved_domains:
                    work_domains[nb] = old_dom
                if l_id in daily_limits:
                    lect_day_count[(l_id, day)] -= 1
                del assignment[cid]
                stats["backtracks"] = int(stats["backtracks"]) + 1

            unassigned.add(cid)
            return None

        return backtrack()

    try:
        if not course_order:
            result: Optional[Dict[str, Choice]] = {}
            stats["nodes"] = 1
        elif any(not d for d in domains.values()):
            stats["nodes"] = 1
            stats["dead_ends"] = int(stats["dead_ends"]) + 1
            result = None
        elif strategy in {"naive", "midterm_heuristic"}:
            result = search_naive_or_midterm(domains)
        else:
            result = solve_final_incremental()

        search_end = perf_counter()
        stats["search_ms"] = (search_end - start) * 1000.0

        if result is not None:
            valid, errors = loaders.validate_assignment(instance, result)
            if not valid:
                raise RuntimeError(f"Solver produced invalid timetable: {errors}")

            try:
                init_score = soft_constraint_score(instance, result)
                stats["initial_soft_score"] = init_score
                stats["final_soft_score"] = init_score
            except ValueError:
                pass

            if strategy == "optimized":
                opt_start = perf_counter()
                result, opt_steps = optimize_timetable(instance, result, domains=domains)
                stats["opt_ms"] = (perf_counter() - opt_start) * 1000.0
                stats["opt_steps"] = opt_steps
                valid, errors = loaders.validate_assignment(instance, result)
                if not valid:
                    raise RuntimeError(f"Optimizer produced invalid timetable: {errors}")
                stats["final_soft_score"] = soft_constraint_score(instance, result)

        if result is not None:
            stats["status"] = "SAT"
        elif limit_hit:
            stats["status"] = "LIMIT_EXCEEDED"
        else:
            stats["status"] = "UNSAT"
        return result
    finally:
        stats["runtime_ms"] = (perf_counter() - start) * 1000.0
        SEARCH_STATS.clear()
        SEARCH_STATS.update(stats)


def soft_constraint_score(instance: dict, assignment: Dict[str, Choice]) -> float:
    """Score a *valid* timetable on optional preferences, 0–100 (higher better).

    Weights each configured soft constraint equally:
    - ``cohort_daily_spread``: mean over cohorts of (distinct days / course count).
    - ``lecturer_load_balance``: 1 - (max_load - min_load) / total_courses.
    """
    valid, errors = loaders.validate_assignment(instance, assignment)
    if not valid:
        raise ValueError("A legal complete timetable is required: " + "; ".join(errors))
    configured = {
        rule["name"] if isinstance(rule, dict) else rule
        for rule in instance["constraints"].get("soft_constraints", [])
    }
    if not configured:
        return 100.0
    unknown = configured - {"cohort_daily_spread", "lecturer_load_balance"}
    if unknown:
        raise ValueError(f"Unsupported soft constraint(s): {sorted(unknown)}")
    scores = []
    if "cohort_daily_spread" in configured:
        group_courses = defaultdict(list)
        for course in instance["courses"]:
            group_courses[course["cohort"]].append(course["course_id"])
        spread = []
        for ids in group_courses.values():
            days = {instance["slot_info"][assignment[cid][2]]["day"] for cid in ids}
            spread.append(len(days) / len(ids))
        scores.append(sum(spread) / len(spread) if spread else 1.0)
    if "lecturer_load_balance" in configured:
        loads = Counter(value[0] for value in assignment.values())
        values = [loads.get(lid, 0) for lid in instance["lecturers"]]
        balance = 1 - (max(values) - min(values)) / max(1, len(assignment)) if values else 1.0
        scores.append(balance)
    return round(100 * sum(scores) / len(scores), 2)


def soft_constraint_breakdown(instance: dict, assignment: Dict[str, Choice]) -> dict:
    """Return component-wise soft-constraint quality metrics for UI and reporting."""
    total = soft_constraint_score(instance, assignment)
    group_courses: Dict[str, List[str]] = defaultdict(list)
    for course in instance["courses"]:
        group_courses[course["cohort"]].append(course["course_id"])

    cohort_days = {}
    spread_ratios = []
    for cohort, ids in sorted(group_courses.items()):
        days = sorted({instance["slot_info"][assignment[cid][2]]["day"] for cid in ids})
        cohort_days[cohort] = {"courses": len(ids), "distinct_days": len(days), "days": days}
        spread_ratios.append(len(days) / len(ids))

    loads = Counter(v[0] for v in assignment.values())
    lecturer_loads = {lid: loads.get(lid, 0) for lid in instance["lecturers"]}
    vals = list(lecturer_loads.values())
    balance = 1.0 - (max(vals) - min(vals)) / max(1, len(assignment)) if vals else 1.0

    return {
        "total_score": total,
        "cohort_daily_spread_pct": round(100.0 * sum(spread_ratios) / max(1, len(spread_ratios)), 2),
        "lecturer_load_balance_pct": round(100.0 * balance, 2),
        "cohort_details": cohort_days,
        "lecturer_loads": lecturer_loads,
    }


def diagnose_unsat(instance: dict) -> List[str]:
    """Provide general, non-hardcoded conflict diagnostics for unsatisfiable instances."""
    notices: List[str] = []
    domains = {c["course_id"]: loaders.candidate_domain(c, instance) for c in instance["courses"]}
    slot_day = {sid: info["day"] for sid, info in instance["slot_info"].items()}

    # 1. Empty individual domain
    for cid, choices in domains.items():
        if not choices:
            notices.append(
                f"Empty domain: course {cid} has 0 eligible (lecturer, room, slot) combinations."
            )

    # 2. Singleton-lecturer slot pigeonhole & daily-limit pigeonhole
    exclusively_taught: Dict[str, List[str]] = defaultdict(list)
    for c in instance["courses"]:
        elig = set(c["eligible_lecturers"])
        if len(elig) == 1:
            exclusively_taught[next(iter(elig))].append(c["course_id"])

    daily_limits = _extract_daily_limits(instance)
    for lecturer, ids in exclusively_taught.items():
        valid_slots = {v[2] for cid in ids for v in domains[cid]}
        if len(ids) > len(valid_slots):
            notices.append(
                f"Lecturer slot pigeonhole: {len(ids)} courses ({', '.join(ids)}) exclusively "
                f"require {lecturer}, but only {len(valid_slots)} distinct valid slot(s) exist "
                f"({', '.join(sorted(valid_slots)) or 'none'})."
            )
        if lecturer in daily_limits:
            valid_days = {slot_day[s] for s in valid_slots}
            cap = len(valid_days) * daily_limits[lecturer]
            if len(ids) > cap:
                notices.append(
                    f"Lecturer daily-limit pigeonhole: {len(ids)} courses ({', '.join(ids)}) "
                    f"exclusively require {lecturer}, whose daily limit ({daily_limits[lecturer]}/day "
                    f"across {len(valid_days)} available day(s)) allows at most {cap} course(s)."
                )

    # 3. Cohort slot pigeonhole
    by_cohort: Dict[str, List[str]] = defaultdict(list)
    for c in instance["courses"]:
        by_cohort[c["cohort"]].append(c["course_id"])
    for cohort, ids in by_cohort.items():
        valid_slots = {v[2] for cid in ids for v in domains[cid]}
        if len(ids) > len(valid_slots):
            notices.append(
                f"Cohort slot pigeonhole: cohort {cohort} has {len(ids)} courses, "
                f"but only {len(valid_slots)} distinct feasible timeslot(s) are available."
            )

    # 4. Room-type total room-slot capacity pigeonhole
    by_rtype: Dict[str, List[str]] = defaultdict(list)
    for c in instance["courses"]:
        by_rtype[c["required_room_type"]].append(c["course_id"])
    for rtype, ids in by_rtype.items():
        room_slots = {(v[1], v[2]) for cid in ids for v in domains[cid]}
        if len(ids) > len(room_slots):
            notices.append(
                f"Room capacity pigeonhole: {len(ids)} {rtype} courses require at least "
                f"{len(ids)} room-slots, but only {len(room_slots)} valid {rtype} room-slot(s) exist."
            )

    if not notices:
        notices.append(
            "Coupled multi-variable conflict: every individual course has a non-empty domain, "
            "but systematic AC-3 / backtracking search exhausted all combinations with no valid solution."
        )
    return notices
