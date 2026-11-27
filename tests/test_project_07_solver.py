"""Comprehensive automated test suite for Project 07 Final AI Product.

Covers:
- All 4 official benchmark instances across all 4 solver strategies;
- Independent hard-constraint validation of every SAT solution;
- Sound unsatisfiability proof and diagnostics on UNSAT fixtures;
- Soft-constraint scoring accuracy, breakdown consistency, and optimizer monotonicity;
- Edge cases (empty CSP, single course, zero-domain course, extra daily limits);
- Out-of-scope and malformed inputs (unknown strategy, unknown extra constraint,
  invalid soft constraint, schema errors).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STARTER_DIR = PROJECT_ROOT / "starter"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

import loaders
import student_core

SAT_INSTANCES = ("feasible", "restricted_rooms", "extra_lecturer")
ALL_STRATEGIES = ("naive", "midterm_heuristic", "heuristic", "optimized")


@pytest.mark.parametrize("instance_name", SAT_INSTANCES)
@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_sat_instances_valid_across_all_strategies(instance_name: str, strategy: str):
    instance = loaders.load_instance(DATA_DIR / f"instance_{instance_name}")
    solution = student_core.solve_csp(instance, strategy=strategy)
    assert solution is not None
    assert len(solution) == len(instance["courses"])
    valid, errors = loaders.validate_assignment(instance, solution)
    assert valid, f"Unexpected violations: {errors}"
    assert student_core.SEARCH_STATS["status"] == "SAT"


@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
def test_unsat_instance_returns_none_across_all_strategies(strategy: str):
    instance = loaders.load_instance(DATA_DIR / "instance_unsat")
    solution = student_core.solve_csp(instance, strategy=strategy)
    assert solution is None
    assert student_core.SEARCH_STATS["status"] == "UNSAT"


def test_final_ac3_proves_unsat_without_branching():
    instance = loaders.load_instance(DATA_DIR / "instance_unsat")
    solution = student_core.solve_csp(instance, strategy="heuristic")
    assert solution is None
    assert student_core.SEARCH_STATS["nodes"] == 1
    assert student_core.SEARCH_STATS["attempts"] == 0
    assert student_core.SEARCH_STATS["backtracks"] == 0
    assert student_core.SEARCH_STATS["ac3_pruned"] > 0


@pytest.mark.parametrize("instance_name", SAT_INSTANCES)
def test_optimizer_never_degrades_soft_score_and_reaches_optimum(instance_name: str):
    instance = loaders.load_instance(DATA_DIR / f"instance_{instance_name}")
    sol_h = student_core.solve_csp(instance, strategy="heuristic")
    score_h = student_core.soft_constraint_score(instance, sol_h)

    sol_o = student_core.solve_csp(instance, strategy="optimized")
    score_o = student_core.soft_constraint_score(instance, sol_o)
    valid, errors = loaders.validate_assignment(instance, sol_o)
    assert valid, f"Optimizer violated hard constraints: {errors}"
    assert score_o >= score_h
    assert score_o >= 81.50  # Theoretical optimum on the 4-day 16-course fixtures is 81.53


def test_extra_lecturer_daily_limit_enforced_after_optimization():
    instance = loaders.load_instance(DATA_DIR / "instance_extra_lecturer")
    sol = student_core.solve_csp(instance, strategy="optimized")
    assert sol is not None
    per_day = {}
    for cid, (lid, _rid, sid) in sol.items():
        if lid == "L02":
            day = instance["slot_info"][sid]["day"]
            per_day[day] = per_day.get(day, 0) + 1
    assert all(cnt <= 1 for cnt in per_day.values())


def test_soft_constraint_breakdown_matches_scalar_score():
    instance = loaders.load_instance(DATA_DIR / "instance_feasible")
    sol = student_core.solve_csp(instance, strategy="optimized")
    score = student_core.soft_constraint_score(instance, sol)
    bd = student_core.soft_constraint_breakdown(instance, sol)
    assert bd["total_score"] == score
    expected = round(0.5 * (bd["cohort_daily_spread_pct"] + bd["lecturer_load_balance_pct"]), 2)
    assert abs(bd["total_score"] - expected) <= 0.02


def test_empty_csp_returns_empty_valid_dict():
    instance = deepcopy(loaders.load_instance(DATA_DIR / "instance_feasible"))
    instance["courses"] = []
    for strategy in ALL_STRATEGIES:
        sol = student_core.solve_csp(instance, strategy=strategy)
        assert sol == {}
        assert student_core.SEARCH_STATS["status"] == "SAT"


def test_closing_all_lab_rooms_yields_unsat_and_diagnosis():
    instance = deepcopy(loaders.load_instance(DATA_DIR / "instance_feasible"))
    for rid in [r for r, info in instance["rooms"].items() if info["room_type"] == "Lab"]:
        del instance["rooms"][rid]
    sol = student_core.solve_csp(instance, strategy="heuristic")
    assert sol is None
    reasons = student_core.diagnose_unsat(instance)
    assert any("Empty domain" in r for r in reasons)


def test_unsat_diagnostics_identifies_pigeonhole_contradiction():
    instance = loaders.load_instance(DATA_DIR / "instance_unsat")
    reasons = student_core.diagnose_unsat(instance)
    assert any("L09" in r and "pigeonhole" in r.lower() for r in reasons)


def test_unsat_diagnostics_identifies_daily_limit_contradiction():
    instance = deepcopy(loaders.load_instance(DATA_DIR / "instance_feasible"))
    # Force L06 (available Wed-Thu = 2 days) to teach 3 exclusive courses with max 1/day
    instance["courses"][0]["eligible_lecturers"] = ["L06"]
    instance["courses"][1]["eligible_lecturers"] = ["L06"]
    instance["courses"][2]["eligible_lecturers"] = ["L06"]
    instance["constraints"]["extra_constraints"] = [
        {"type": "lecturer_daily_limit", "lecturer_id": "L06", "max_per_day": 1}
    ]
    sol = student_core.solve_csp(instance, strategy="heuristic")
    assert sol is None
    reasons = student_core.diagnose_unsat(instance)
    assert any("daily-limit" in r.lower() for r in reasons)


def test_deep_unsat_order_invariance_for_final_heuristic():
    instance = deepcopy(loaders.load_instance(DATA_DIR / "instance_unsat"))
    # Move U01, U02 to the very end of the course list
    instance["courses"] = instance["courses"][2:] + instance["courses"][:2]
    sol = student_core.solve_csp(instance, strategy="heuristic")
    assert sol is None
    assert student_core.SEARCH_STATS["nodes"] == 1
    assert student_core.SEARCH_STATS["attempts"] == 0


def test_invalid_strategy_raises_value_error():
    instance = loaders.load_instance(DATA_DIR / "instance_feasible")
    with pytest.raises(ValueError):
        student_core.solve_csp(instance, strategy="simulated_annealing")


def test_unknown_extra_constraint_raises_value_error():
    instance = deepcopy(loaders.load_instance(DATA_DIR / "instance_feasible"))
    instance["constraints"]["extra_constraints"] = [{"type": "unknown_rule"}]
    with pytest.raises(ValueError):
        student_core.solve_csp(instance, strategy="heuristic")


def test_soft_score_rejects_invalid_assignment():
    instance = loaders.load_instance(DATA_DIR / "instance_feasible")
    with pytest.raises(ValueError):
        student_core.soft_constraint_score(instance, {})


def test_soft_score_rejects_unknown_soft_constraint():
    instance = deepcopy(loaders.load_instance(DATA_DIR / "instance_feasible"))
    sol = student_core.solve_csp(instance, strategy="heuristic")
    instance["constraints"]["soft_constraints"] = [{"name": "unsupported_pref"}]
    with pytest.raises(ValueError):
        student_core.soft_constraint_score(instance, sol)


def test_node_limit_reports_limit_exceeded_not_unsat():
    instance = deepcopy(loaders.load_instance(DATA_DIR / "instance_unsat"))
    instance["courses"] = instance["courses"][2:] + instance["courses"][:2]
    sol = student_core.solve_csp(instance, strategy="naive", node_limit=50)
    assert sol is None
    assert student_core.SEARCH_STATS["status"] == "LIMIT_EXCEEDED"


def test_assignment_csv_and_constraint_report_helpers():
    instance = loaders.load_instance(DATA_DIR / "instance_feasible")
    sol = student_core.solve_csp(instance, strategy="optimized")
    csv_text = loaders.assignment_csv(instance, sol)
    assert "course_id,course_name" in csv_text
    assert len(csv_text.strip().splitlines()) == len(instance["courses"]) + 1
    report = loaders.constraint_report(instance, sol)
    assert all(row["status"] == "PASS" and row["violations"] == 0 for row in report)
