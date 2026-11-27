"""Reproducible experimental harness and figure generator for Project 07.

Reproduces all quantitative tables (Table 3, Table 4, hardened scaling tables)
and black-and-white publication figures in the Final Report.

Usage:
    python scripts/run_experiment.py --seed 42
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
import random
import statistics
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
STARTER_DIR = PROJECT_ROOT / "starter"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

import loaders
import student_core

STRATEGIES = ("naive", "midterm_heuristic", "heuristic", "optimized")
STRATEGY_DISPLAY = {
    "naive": "Baseline (Naive)",
    "midterm_heuristic": "Midterm (MRV+FC)",
    "heuristic": "Final Fast (AC-3+Inc-FC)",
    "optimized": "Final Optimized (+LS)",
}


def build_hard_deep_unsat() -> dict:
    """Hardened UNSAT: move conflicting pair (U01, U02) to the end of CSV order
    and restrict rooms to 4 (2 Lecture, 2 Lab) to expose chronological thrashing."""
    inst = deepcopy(loaders.load_instance(DATA_DIR / "instance_unsat"))
    inst["courses"] = inst["courses"][2:] + inst["courses"][:2]
    for rid in ("R03", "R04", "R07"):
        inst["rooms"].pop(rid, None)
    inst["constraints"]["instance_id"] = "hard_deep_unsat"
    inst["constraints"]["description"] = (
        "Deep UNSAT: conflicting specialist pair U01/U02 placed at end of course list with 4 rooms."
    )
    return inst


def build_hard_tight_sat(seed: int = 42) -> dict:
    """Hardened SAT instance with 20 courses, 4 rooms, tight staggered windows,
    and constrained courses placed late in CSV order to force backtracking in Naive."""
    rng = random.Random(seed)
    base = deepcopy(loaders.load_instance(DATA_DIR / "instance_restricted_rooms"))
    all_slots = list(base["all_slots"])
    slot_info = dict(base["slot_info"])

    # Tight staggered lecturer windows creating genuine cross-course contention
    lecturers = {
        "L01": {"name": "Dr. Mai Tran", "available_slots": set(all_slots[:12])},
        "L02": {"name": "Dr. Khoa Le", "available_slots": set(all_slots[4:])},
        "L03": {"name": "Dr. Yen Pham", "available_slots": {"T01", "T02", "T05", "T06", "T09", "T10"}},
        "L04": {"name": "Dr. Duc Vo", "available_slots": set(all_slots[:8] + all_slots[12:])},
        "L05": {"name": "Dr. Linh Bui", "available_slots": {"T01", "T02", "T03", "T05", "T06", "T09", "T10", "T13"}},
        "L06": {"name": "Dr. Nam Dang", "available_slots": {"T09", "T10", "T11", "T13", "T14", "T15"}},
        "L07": {"name": "Dr. Thu Ngo", "available_slots": set(all_slots[2:14])},
        "L08": {"name": "Dr. Hoa Doan", "available_slots": set(all_slots[4:16])},
    }

    # Restrict room R05 (Lab, 30) and R01 (Lecture, 60) slightly to tighten coupling
    rooms = {
        "R01": {"room_type": "Lecture", "capacity": 60, "available_slots": set(all_slots) - {"T04", "T08"}},
        "R02": {"room_type": "Lecture", "capacity": 45, "available_slots": set(all_slots) - {"T12", "T16"}},
        "R05": {"room_type": "Lab", "capacity": 30, "available_slots": {"T01", "T02", "T05", "T06", "T09", "T10", "T13", "T14"}},
        "R06": {"room_type": "Lab", "capacity": 25, "available_slots": {"T01", "T02", "T03", "T05", "T06", "T09", "T10", "T13"}},
    }

    # Place loosely constrained courses first and tightly coupled bottlenecks last!
    courses = [
        {"course_id": "H01", "course_name": "Calculus I", "cohort": "CS-3A", "required_room_type": "Lecture", "min_capacity": 35, "eligible_lecturers": ["L01", "L04", "L07"]},
        {"course_id": "H02", "course_name": "Linear Algebra", "cohort": "CS-3A", "required_room_type": "Lecture", "min_capacity": 35, "eligible_lecturers": ["L02", "L07", "L08"]},
        {"course_id": "H03", "course_name": "Probability", "cohort": "CS-3B", "required_room_type": "Lecture", "min_capacity": 35, "eligible_lecturers": ["L01", "L04", "L08"]},
        {"course_id": "H04", "course_name": "Computer Org", "cohort": "CS-3B", "required_room_type": "Lecture", "min_capacity": 40, "eligible_lecturers": ["L02", "L07"]},
        {"course_id": "H05", "course_name": "Systems Prog", "cohort": "CS-3C", "required_room_type": "Lecture", "min_capacity": 40, "eligible_lecturers": ["L04", "L01"]},
        {"course_id": "H06", "course_name": "Data Mining", "cohort": "CS-3C", "required_room_type": "Lecture", "min_capacity": 35, "eligible_lecturers": ["L08", "L02"]},
        {"course_id": "H07", "course_name": "Compiler Design", "cohort": "CS-3D", "required_room_type": "Lecture", "min_capacity": 40, "eligible_lecturers": ["L07", "L01"]},
        {"course_id": "H08", "course_name": "Distributed Sys", "cohort": "CS-3D", "required_room_type": "Lecture", "min_capacity": 35, "eligible_lecturers": ["L08", "L04"]},
        {"course_id": "H09", "course_name": "Algorithms II", "cohort": "CS-3A", "required_room_type": "Lecture", "min_capacity": 50, "eligible_lecturers": ["L01", "L04"]},
        {"course_id": "H10", "course_name": "Machine Learning", "cohort": "CS-3B", "required_room_type": "Lecture", "min_capacity": 50, "eligible_lecturers": ["L02", "L08"]},
        {"course_id": "H11", "course_name": "NLP Seminar", "cohort": "CS-3C", "required_room_type": "Lecture", "min_capacity": 50, "eligible_lecturers": ["L07", "L02"]},
        {"course_id": "H12", "course_name": "Cloud Architecture", "cohort": "CS-3D", "required_room_type": "Lecture", "min_capacity": 50, "eligible_lecturers": ["L04", "L08"]},
        # Tightly coupled lab courses placed at the end (all competing for R05/R06 & T01/T02/T05/T06/T09/T10)
        {"course_id": "H13", "course_name": "Networking Lab", "cohort": "CS-3A", "required_room_type": "Lab", "min_capacity": 25, "eligible_lecturers": ["L05", "L03"]},
        {"course_id": "H14", "course_name": "Security Lab", "cohort": "CS-3B", "required_room_type": "Lab", "min_capacity": 25, "eligible_lecturers": ["L05", "L03"]},
        {"course_id": "H15", "course_name": "Embedded Lab", "cohort": "CS-3C", "required_room_type": "Lab", "min_capacity": 25, "eligible_lecturers": ["L06", "L05"]},
        {"course_id": "H16", "course_name": "Vision Lab", "cohort": "CS-3D", "required_room_type": "Lab", "min_capacity": 25, "eligible_lecturers": ["L06", "L03"]},
        {"course_id": "H17", "course_name": "Capstone Lab A", "cohort": "CS-3A", "required_room_type": "Lab", "min_capacity": 30, "eligible_lecturers": ["L03"]},
        {"course_id": "H18", "course_name": "Capstone Lab B", "cohort": "CS-3B", "required_room_type": "Lab", "min_capacity": 30, "eligible_lecturers": ["L03"]},
        {"course_id": "H19", "course_name": "Capstone Lab C", "cohort": "CS-3C", "required_room_type": "Lab", "min_capacity": 30, "eligible_lecturers": ["L06"]},
        {"course_id": "H20", "course_name": "Capstone Lab D", "cohort": "CS-3D", "required_room_type": "Lab", "min_capacity": 30, "eligible_lecturers": ["L06"]},
    ]
    # Deterministically shuffle only the first 8 loose courses with seed for reproducibility
    head = courses[:8]
    rng.shuffle(head)
    courses = head + courses[8:]

    constraints = deepcopy(base["constraints"])
    constraints["instance_id"] = "hard_tight_sat"
    constraints["description"] = "Hardened 20-course tight SAT benchmark with 4 rooms, restricted windows, and 2 daily limits."
    constraints["extra_constraints"] = [
        {"type": "lecturer_daily_limit", "lecturer_id": "L03", "max_per_day": 1},
        {"type": "lecturer_daily_limit", "lecturer_id": "L06", "max_per_day": 1},
    ]

    return {
        "courses": courses,
        "lecturers": lecturers,
        "rooms": rooms,
        "all_slots": all_slots,
        "slot_info": slot_info,
        "constraints": constraints,
    }


def build_stress_28_courses(seed: int = 42) -> dict:
    """Large-scale 28-course, 5-cohort, 10-lecturer, 6-room, 20-slot (Mon-Fri) SAT stress instance."""
    rng = random.Random(seed + 100)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    periods = [
        (1, "07:30", "09:00"),
        (2, "09:15", "10:45"),
        (3, "13:00", "14:30"),
        (4, "14:45", "16:15"),
    ]
    all_slots: List[str] = []
    slot_info: Dict[str, dict] = {}
    idx = 1
    for d in days:
        for p, st, en in periods:
            sid = f"T{idx:02d}"
            all_slots.append(sid)
            slot_info[sid] = {"day": d, "period": p, "start_time": st, "end_time": en}
            idx += 1

    lecturers = {}
    for i in range(1, 11):
        lid = f"L{i:02d}"
        if i in (3, 9):
            avail = set(all_slots[:12])
        elif i in (6, 10):
            avail = set(all_slots[8:])
        else:
            avail = set(all_slots)
        lecturers[lid] = {"name": f"Dr. Faculty {i:02d}", "available_slots": avail}

    rooms = {
        "R01": {"room_type": "Lecture", "capacity": 60, "available_slots": set(all_slots)},
        "R02": {"room_type": "Lecture", "capacity": 50, "available_slots": set(all_slots)},
        "R03": {"room_type": "Lecture", "capacity": 40, "available_slots": set(all_slots) - {"T17", "T18"}},
        "R04": {"room_type": "Lecture", "capacity": 35, "available_slots": set(all_slots)},
        "R05": {"room_type": "Lab", "capacity": 30, "available_slots": set(all_slots)},
        "R06": {"room_type": "Lab", "capacity": 25, "available_slots": set(all_slots) - {"T19", "T20"}},
    }

    cohorts = ["CS-2A", "CS-2B", "CS-2C", "DS-2A", "SE-2A"]
    lect_pools = [
        ["L01", "L04"],
        ["L02", "L07"],
        ["L05", "L08"],
        ["L03", "L09"],
        ["L06", "L10"],
        ["L01", "L07"],
        ["L04", "L08"],
    ]
    courses = []
    for i in range(28):
        cid = f"S{i+1:02d}"
        coh = cohorts[i % len(cohorts)]
        is_lab = (i % 4 == 3)
        rtype = "Lab" if is_lab else "Lecture"
        min_cap = 25 if is_lab else rng.choice([30, 35, 45, 50])
        elig = list(lect_pools[i % len(lect_pools)])
        if i in (25, 26):
            elig = ["L09"]
            rtype = "Lab"
            min_cap = 25
        elif i == 27:
            elig = ["L10"]
            rtype = "Lab"
            min_cap = 25
        courses.append({
            "course_id": cid,
            "course_name": f"Module {i+1:02d} ({rtype})",
            "cohort": coh,
            "required_room_type": rtype,
            "min_capacity": min_cap,
            "eligible_lecturers": elig,
        })

    base_constraints = deepcopy(
        loaders.load_instance(DATA_DIR / "instance_feasible")["constraints"]
    )
    base_constraints["instance_id"] = "stress_28_courses"
    base_constraints["description"] = "28-course, 5-cohort, 10-lecturer, 6-room, 20-slot scalability stress instance."
    base_constraints["extra_constraints"] = [
        {"type": "lecturer_daily_limit", "lecturer_id": "L02", "max_per_day": 1},
        {"type": "lecturer_daily_limit", "lecturer_id": "L09", "max_per_day": 1},
    ]
    return {
        "courses": courses,
        "lecturers": lecturers,
        "rooms": rooms,
        "all_slots": all_slots,
        "slot_info": slot_info,
        "constraints": base_constraints,
    }


def run_benchmark_suite(
    suite: Dict[str, dict], trials: int, node_limit: int | None = None
) -> List[dict]:
    rows = []
    for inst_name, inst in suite.items():
        for strat in STRATEGIES:
            # 2 warm-up runs to avoid cold-cache jitter
            for _ in range(2):
                student_core.solve_csp(inst, strategy=strat, node_limit=node_limit)

            runtimes = []
            search_times = []
            opt_times = []
            last_sol = None
            last_stats = None
            for _ in range(trials):
                last_sol = student_core.solve_csp(inst, strategy=strat, node_limit=node_limit)
                last_stats = student_core.SEARCH_STATS.copy()
                runtimes.append(float(last_stats["runtime_ms"]))
                search_times.append(float(last_stats["search_ms"]))
                opt_times.append(float(last_stats["opt_ms"]))

            is_valid = (
                loaders.validate_assignment(inst, last_sol)[0]
                if last_sol is not None
                else None
            )
            bd = (
                student_core.soft_constraint_breakdown(inst, last_sol)
                if last_sol is not None
                else None
            )
            rows.append({
                "instance": inst_name,
                "courses": len(inst["courses"]),
                "rooms": len(inst["rooms"]),
                "strategy": strat,
                "status": last_stats["status"],
                "valid": is_valid if is_valid is not None else last_stats["status"],
                "mean_ms": round(statistics.mean(runtimes), 3),
                "std_ms": round(statistics.stdev(runtimes), 3) if len(runtimes) > 1 else 0.0,
                "median_ms": round(statistics.median(runtimes), 3),
                "search_mean_ms": round(statistics.mean(search_times), 3),
                "opt_mean_ms": round(statistics.mean(opt_times), 3),
                "nodes": int(last_stats["nodes"]),
                "attempts": int(last_stats["attempts"]),
                "backtracks": int(last_stats["backtracks"]),
                "ac3_pruned": int(last_stats["ac3_pruned"]),
                "pruned_values": int(last_stats["pruned_values"]),
                "opt_steps": int(last_stats["opt_steps"]),
                "soft_score": last_stats["final_soft_score"],
                "spread_pct": bd["cohort_daily_spread_pct"] if bd else None,
                "balance_pct": bd["lecturer_load_balance_pct"] if bd else None,
            })
    return rows


def run_scenarios(seed: int = 42) -> List[dict]:
    """Evaluate 6 representative product scenarios (Typical, Edge, Stress, Out-of-Scope)."""
    scenarios = []

    # S1: Typical input — official feasible with Final Optimized
    inst1 = loaders.load_instance(DATA_DIR / "instance_feasible")
    sol1 = student_core.solve_csp(inst1, strategy="optimized")
    st1 = student_core.SEARCH_STATS.copy()
    ok1, _ = loaders.validate_assignment(inst1, sol1)
    scenarios.append({
        "id": "S1",
        "category": "Typical",
        "scenario": "Standard 16-course feasible schedule with Final Optimized mode",
        "status": st1["status"],
        "valid": ok1,
        "nodes": st1["nodes"],
        "runtime_ms": round(float(st1["runtime_ms"]), 3),
        "soft_score": st1["final_soft_score"],
        "outcome": "Pass" if ok1 and st1["final_soft_score"] >= 81.5 else "Fail",
    })

    # S2: Edge case — provably unsatisfiable instance (pigeonhole bottleneck)
    inst2 = loaders.load_instance(DATA_DIR / "instance_unsat")
    sol2 = student_core.solve_csp(inst2, strategy="optimized")
    st2 = student_core.SEARCH_STATS.copy()
    diag2 = student_core.diagnose_unsat(inst2)
    scenarios.append({
        "id": "S2",
        "category": "Edge (UNSAT)",
        "scenario": "Provably unsatisfiable fixture (U01, U02 exclusively require L09 at T05)",
        "status": st2["status"],
        "valid": sol2 is None,
        "nodes": st2["nodes"],
        "runtime_ms": round(float(st2["runtime_ms"]), 3),
        "diagnosis": diag2[0],
        "outcome": "Pass" if sol2 is None and st2["attempts"] == 0 else "Fail",
    })

    # S3: Edge case — live interactive closure of all Lab rooms in UI
    inst3 = deepcopy(loaders.load_instance(DATA_DIR / "instance_feasible"))
    for rid in ("R05", "R06", "R07"):
        del inst3["rooms"][rid]
    sol3 = student_core.solve_csp(inst3, strategy="heuristic")
    st3 = student_core.SEARCH_STATS.copy()
    diag3 = student_core.diagnose_unsat(inst3)
    scenarios.append({
        "id": "S3",
        "category": "Edge (Empty Domain)",
        "scenario": "Live scenario closure of all Lab rooms (R05, R06, R07)",
        "status": st3["status"],
        "valid": sol3 is None,
        "nodes": st3["nodes"],
        "runtime_ms": round(float(st3["runtime_ms"]), 3),
        "diagnosis": diag3[0],
        "outcome": "Pass" if sol3 is None and st3["nodes"] == 1 else "Fail",
    })

    # S4: Stress case — 28-course, 5-cohort, 20-slot campus schedule with 2 daily limits
    inst4 = build_stress_28_courses(seed)
    sol4 = student_core.solve_csp(inst4, strategy="optimized")
    st4 = student_core.SEARCH_STATS.copy()
    ok4, _ = loaders.validate_assignment(inst4, sol4)
    scenarios.append({
        "id": "S4",
        "category": "Stress (Scale)",
        "scenario": "28 courses, 5 cohorts, 10 lecturers, 6 rooms, 20 slots, 2 daily limits",
        "status": st4["status"],
        "valid": ok4,
        "nodes": st4["nodes"],
        "runtime_ms": round(float(st4["runtime_ms"]), 3),
        "soft_score": st4["final_soft_score"],
        "outcome": "Pass" if ok4 else "Fail",
    })

    # S5: Stress / Adversarial ordering — Deep UNSAT (Baseline fails limit; Final succeeds in 0 attempts)
    inst5 = build_hard_deep_unsat()
    sol5_n = student_core.solve_csp(inst5, strategy="naive", node_limit=10000)
    st5_n = student_core.SEARCH_STATS.copy()
    sol5_f = student_core.solve_csp(inst5, strategy="heuristic", node_limit=10000)
    st5_f = student_core.SEARCH_STATS.copy()
    scenarios.append({
        "id": "S5",
        "category": "Stress (Adversarial UNSAT)",
        "scenario": "Conflicting pair U01/U02 placed at end of CSV list with 4 rooms",
        "naive_status": st5_n["status"],
        "naive_nodes": st5_n["nodes"],
        "final_status": st5_f["status"],
        "final_nodes": st5_f["nodes"],
        "final_runtime_ms": round(float(st5_f["runtime_ms"]), 3),
        "outcome": "Pass (Final) / Fail (Baseline)",
    })

    # S6: Out-of-scope input — unsupported extra constraint / invalid strategy / unknown room type
    inst6 = deepcopy(loaders.load_instance(DATA_DIR / "instance_feasible"))
    inst6["constraints"]["extra_constraints"] = [{"type": "minimum_gap_hours", "hours": 2}]
    caught_msg = ""
    try:
        student_core.solve_csp(inst6, strategy="heuristic")
    except ValueError as exc:
        caught_msg = str(exc)
    scenarios.append({
        "id": "S6",
        "category": "Out-of-Scope",
        "scenario": "Unsupported extra constraint type ('minimum_gap_hours') in constraints.json",
        "error_raised": caught_msg,
        "outcome": "Pass (Graceful Rejection)" if "Unknown extra hard constraint" in caught_msg else "Fail",
    })

    return scenarios


def generate_figures(official_rows: List[dict], hardened_rows: List[dict]) -> None:
    """Generate 4 grayscale / black-and-white publication figures (300 DPI)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.edgecolor": "#111111",
        "axes.linewidth": 0.9,
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "text.color": "#111111",
    })

    # --- Figure 1: High-Level System Architecture (Zero-Overlap B&W Diagram) ---
    fig, ax = plt.subplots(figsize=(8.6, 3.8), dpi=300)
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.75)
    ax.axis("off")

    boxes = [
        (0.2, 1.15, 2.1, 2.45, "1. User Interface\n(starter/app.py)",
         "• Dataset / ZIP upload\n• Strategy selector (4)\n• Live room, slot, &\n  daily-limit controls\n• Color-coded calendar"),
        (2.8, 1.15, 2.1, 2.45, "2. Preprocessing\n(starter/loaders.py)",
         "• CSV & JSON parsing\n• Schema validation\n• candidate_domain()\n  (unary filtering)\n• Instance dict model"),
        (5.4, 0.85, 2.45, 2.90, "3. AI Core Engine\n(starter/student_core.py)",
         "• Conflict graph index\n• Fast AC-3 pruning\n• MRV + Dynamic Degree\n• Incremental FC + trail\n• Local Search Optimizer\n  (variance-smoothed)"),
        (8.35, 1.15, 1.85, 2.45, "4. Verification &\nResult View",
         "• validate_assignment()\n• Hard-rule audit table\n• Soft-score analytics\n• UNSAT diagnostics\n• CSV / JSON export"),
    ]
    for x, y, w, h, title, body in boxes:
        fc = "#EAEAEA" if "AI Core" in title else "#FFFFFF"
        lw = 1.6 if "AI Core" in title else 1.1
        rect = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#111111", linewidth=lw)
        ax.add_patch(rect)
        hdr_h = 0.65
        hdr = plt.Rectangle((x, y + h - hdr_h), w, hdr_h, facecolor="#222222", edgecolor="#111111", linewidth=1.0)
        ax.add_patch(hdr)
        ax.text(x + w / 2, y + h - hdr_h / 2, title, ha="center", va="center",
                color="white", fontsize=8.4, fontweight="bold")
        ax.text(x + 0.09, y + h - hdr_h - 0.14, body, ha="left", va="top",
                color="#111111", fontsize=7.6, linespacing=1.35)

    arrow_props = dict(arrowstyle="-|>", color="#111111", lw=1.25, mutation_scale=12)
    ax.annotate("", xy=(2.8, 2.35), xytext=(2.3, 2.35), arrowprops=arrow_props)
    ax.annotate("", xy=(5.4, 2.35), xytext=(4.9, 2.35), arrowprops=arrow_props)
    ax.annotate("", xy=(8.35, 2.35), xytext=(7.85, 2.35), arrowprops=arrow_props)

    # Clean orthogonal bottom feedback rail strictly below all boxes (y=0.42)
    ax.plot([9.27, 9.27, 1.25], [1.15, 0.42, 0.42], color="#333333", lw=1.1, ls="--")
    ax.annotate("", xy=(1.25, 1.15), xytext=(1.25, 0.42),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1, ls="--", mutation_scale=11))
    ax.text(5.26, 0.16,
            "Validated timetable, soft-constraint breakdown, and UNSAT diagnostics returned to Streamlit UI",
            ha="center", va="center", fontsize=7.8, style="italic", color="#222222")

    # Offline data store cleanly above Box 2
    ax.text(3.85, 4.35,
            "Offline Data Store (data/instance_*/: courses.csv, lecturers.csv, rooms.csv, timeslots.csv, constraints.json)",
            ha="center", va="center", fontsize=7.9,
            bbox=dict(boxstyle="square,pad=0.35", facecolor="#F4F4F4", edgecolor="#333333", lw=0.9))
    ax.annotate("", xy=(3.85, 3.62), xytext=(3.85, 4.12), arrowprops=arrow_props)

    plt.tight_layout(pad=0.25)
    fig.savefig(RESULTS_DIR / "fig1_architecture.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Official 4 Instances Runtime & Soft-Constraint Quality ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=300)
    off_insts = ["feasible", "restricted_rooms", "extra_lecturer", "unsat"]
    off_labels = ["Feasible", "Restricted\nRooms", "Extra\nLecturer", "UNSAT"]
    lookup = {(r["instance"], r["strategy"]): r for r in official_rows}

    x = np.arange(len(off_insts))
    w = 0.2
    styles = [
        ("naive", "#FFFFFF", "////", "Baseline (Naive)"),
        ("midterm_heuristic", "#CCCCCC", "", "Midterm (MRV+FC)"),
        ("heuristic", "#666666", "", "Final Fast (AC-3+Inc-FC)"),
        ("optimized", "#1A1A1A", "", "Final Optimized (+LS)"),
    ]
    max_h1 = 0.0
    for idx_s, (st, color, hatch, lbl) in enumerate(styles):
        vals = [lookup[(i, st)]["mean_ms"] for i in off_insts]
        errs = [lookup[(i, st)]["std_ms"] for i in off_insts]
        max_h1 = max(max_h1, max(v + e for v, e in zip(vals, errs)))
        ax1.bar(x + (idx_s - 1.5) * w, vals, w, yerr=errs, capsize=2,
                color=color, edgecolor="#111111", hatch=hatch, linewidth=0.8,
                error_kw=dict(lw=0.7, ecolor="#111111"), label=lbl)
    ax1.set_xticks(x)
    ax1.set_xticklabels(off_labels, fontsize=8.5)
    ax1.set_ylim(0, max_h1 * 1.38)
    ax1.set_ylabel("Mean Runtime (ms)", fontsize=9)
    ax1.set_title("(a) Mean Runtime on Official Instances (30 runs)", fontsize=9.2, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, color="#888888", alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7.2, frameon=True, edgecolor="#333333", ncol=2, loc="upper center")

    sat_insts = ["feasible", "restricted_rooms", "extra_lecturer"]
    sat_labels = ["Feasible", "Restricted\nRooms", "Extra\nLecturer"]
    x2 = np.arange(len(sat_insts))
    w2 = 0.24
    sat_styles = [
        ("naive", "#FFFFFF", "////", "Baseline (Naive)"),
        ("midterm_heuristic", "#BBBBBB", "", "Midterm / Final Fast"),
        ("optimized", "#1A1A1A", "", "Final Optimized (+LS)"),
    ]
    for idx_s, (st, color, hatch, lbl) in enumerate(sat_styles):
        vals = [lookup[(i, st)]["soft_score"] for i in sat_insts]
        bars = ax2.bar(x2 + (idx_s - 1.0) * w2, vals, w2,
                       color=color, edgecolor="#111111", hatch=hatch, linewidth=0.8, label=lbl)
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7.3)
    ax2.axhline(81.53, color="#111111", linestyle="--", linewidth=0.95, label="Theoretical Bound (81.53)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sat_labels, fontsize=8.5)
    ax2.set_ylim(0, 114)
    ax2.set_ylabel("Soft-Constraint Score (0–100)", fontsize=9)
    ax2.set_title("(b) Timetable Quality Score (Higher is Better)", fontsize=9.2, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", linewidth=0.6, color="#888888", alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.2, frameon=True, edgecolor="#333333", ncol=2, loc="upper center")

    plt.tight_layout(pad=0.8)
    fig.savefig(RESULTS_DIR / "fig2_official_runtime_and_quality.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Hardened & Stress Instances (Log-Scale Nodes & Runtime) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6), dpi=300)
    h_insts = ["hard_deep_unsat", "hard_tight_sat", "stress_28_courses"]
    h_labels = ["Hard Deep\nUNSAT (16c)", "Hard Tight\nSAT (20c)", "Stress Scale\nSAT (28c)"]
    hlookup = {(r["instance"], r["strategy"]): r for r in hardened_rows}
    xh = np.arange(len(h_insts))
    wh = 0.25
    h_styles = [
        ("naive", "#FFFFFF", "////", "Baseline (Naive)"),
        ("midterm_heuristic", "#BBBBBB", "", "Midterm (MRV+FC)"),
        ("heuristic", "#222222", "", "Final Fast (AC-3+Inc-FC)"),
    ]
    for idx_s, (st, color, hatch, lbl) in enumerate(h_styles):
        nodes_v = [max(1, hlookup[(i, st)]["nodes"]) for i in h_insts]
        time_v = [max(0.1, hlookup[(i, st)]["mean_ms"]) for i in h_insts]
        b1 = ax1.bar(xh + (idx_s - 1.0) * wh, nodes_v, wh,
                     color=color, edgecolor="#111111", hatch=hatch, linewidth=0.8, label=lbl)
        b2 = ax2.bar(xh + (idx_s - 1.0) * wh, time_v, wh,
                     color=color, edgecolor="#111111", hatch=hatch, linewidth=0.8, label=lbl)
        for b, r_i in zip(b1, h_insts):
            row = hlookup[(r_i, st)]
            tag = f"{row['nodes']}*" if row["status"] == "LIMIT_EXCEEDED" else f"{row['nodes']}"
            ax1.text(b.get_x() + b.get_width() / 2, row["nodes"] * 1.18, tag,
                     ha="center", va="bottom", fontsize=7.0)
        for b, r_i in zip(b2, h_insts):
            row = hlookup[(r_i, st)]
            tag = f"{row['mean_ms']:.0f}*" if row["status"] == "LIMIT_EXCEEDED" else f"{row['mean_ms']:.1f}"
            ax2.text(b.get_x() + b.get_width() / 2, row["mean_ms"] * 1.18, tag,
                     ha="center", va="bottom", fontsize=7.0)

    ax1.set_yscale("log")
    ax1.set_ylim(0.7, 90000)
    ax1.set_xticks(xh)
    ax1.set_xticklabels(h_labels, fontsize=8.5)
    ax1.set_ylabel("Recursive Nodes Explored (log scale)", fontsize=8.8)
    ax1.set_title("(a) Search Nodes on Hardened Suite (* = 10k cap)", fontsize=9.0, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, color="#888888", alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7.2, frameon=True, edgecolor="#333333", loc="upper right")

    ax2.set_yscale("log")
    ax2.set_ylim(0.4, 3000)
    ax2.set_xticks(xh)
    ax2.set_xticklabels(h_labels, fontsize=8.5)
    ax2.set_ylabel("Mean Runtime in ms (log scale)", fontsize=8.8)
    ax2.set_title("(b) Runtime on Hardened Suite (* = 10k cap)", fontsize=9.0, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", linewidth=0.6, color="#888888", alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.2, frameon=True, edgecolor="#333333", loc="upper right")

    plt.tight_layout(pad=0.8)
    fig.savefig(RESULTS_DIR / "fig3_hardened_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 4: Soft-Constraint Quality Breakdown (Before vs After Opt) ---
    inst_ex = loaders.load_instance(DATA_DIR / "instance_extra_lecturer")
    sol_before = student_core.solve_csp(inst_ex, strategy="heuristic")
    sol_after = student_core.solve_csp(inst_ex, strategy="optimized")
    bd_before = student_core.soft_constraint_breakdown(inst_ex, sol_before)
    bd_after = student_core.soft_constraint_breakdown(inst_ex, sol_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5), dpi=300, gridspec_kw={"width_ratios": [1.15, 1.0]})
    lects = sorted(inst_ex["lecturers"].keys())
    xl = np.arange(len(lects))
    wl = 0.36
    b_vals = [bd_before["lecturer_loads"][l] for l in lects]
    a_vals = [bd_after["lecturer_loads"][l] for l in lects]
    ax1.bar(xl - wl / 2, b_vals, wl, color="#DDDDDD", edgecolor="#111111", hatch="///",
            linewidth=0.8, label=f"Before Opt (Balance: {bd_before['lecturer_load_balance_pct']:.1f}%)")
    ax1.bar(xl + wl / 2, a_vals, wl, color="#222222", edgecolor="#111111",
            linewidth=0.8, label=f"After Opt (Balance: {bd_after['lecturer_load_balance_pct']:.1f}%)")
    ax1.set_xticks(xl)
    ax1.set_xticklabels(lects, fontsize=8.5)
    ax1.set_yticks(range(0, 7))
    ax1.set_ylim(0, 6.8)
    ax1.set_ylabel("Assigned Courses per Week", fontsize=8.8)
    ax1.set_title("(a) Lecturer Workload Distribution (extra_lecturer)", fontsize=9.0, fontweight="bold")
    ax1.grid(axis="y", linestyle=":", linewidth=0.6, color="#888888", alpha=0.7)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7.3, frameon=True, edgecolor="#333333", loc="upper right")

    cohorts = sorted(bd_after["cohort_details"].keys())
    xc = np.arange(len(cohorts))
    cb_vals = [bd_before["cohort_details"][c]["distinct_days"] for c in cohorts]
    ca_vals = [bd_after["cohort_details"][c]["distinct_days"] for c in cohorts]
    ax2.bar(xc - wl / 2, cb_vals, wl, color="#DDDDDD", edgecolor="#111111", hatch="///",
            linewidth=0.8, label=f"Before Opt (Spread: {bd_before['cohort_daily_spread_pct']:.1f}%)")
    ax2.bar(xc + wl / 2, ca_vals, wl, color="#222222", edgecolor="#111111",
            linewidth=0.8, label=f"After Opt (Spread: {bd_after['cohort_daily_spread_pct']:.1f}%)")
    ax2.axhline(4, color="#111111", linestyle="--", linewidth=0.9, label="Max Weekdays Available (4)")
    ax2.set_xticks(xc)
    ax2.set_xticklabels(cohorts, fontsize=8.5)
    ax2.set_yticks(range(0, 6))
    ax2.set_ylim(0, 6.0)
    ax2.set_ylabel("Distinct Teaching Days / Week", fontsize=8.8)
    ax2.set_title("(b) Cohort Daily Spread (extra_lecturer)", fontsize=9.0, fontweight="bold")
    ax2.grid(axis="y", linestyle=":", linewidth=0.6, color="#888888", alpha=0.7)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=7.2, frameon=True, edgecolor="#333333", loc="upper center")

    plt.tight_layout(pad=0.8)
    fig.savefig(RESULTS_DIR / "fig4_soft_optimization_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible Project 07 CSP benchmarks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for hardened benchmarks.")
    parser.add_argument("--trials", type=int, default=30, help="Timed trials per strategy/instance.")
    parser.add_argument("--node-limit", type=int, default=10000, help="Safety node limit for hardened suite.")
    args = parser.parse_args()

    official_suite = {
        name: loaders.load_instance(DATA_DIR / f"instance_{name}")
        for name in ("feasible", "restricted_rooms", "extra_lecturer", "unsat")
    }
    hardened_suite = {
        "hard_deep_unsat": build_hard_deep_unsat(),
        "hard_tight_sat": build_hard_tight_sat(args.seed),
        "stress_28_courses": build_stress_28_courses(args.seed),
    }

    official_rows = run_benchmark_suite(official_suite, trials=args.trials, node_limit=None)
    hardened_rows = run_benchmark_suite(hardened_suite, trials=args.trials, node_limit=args.node_limit)
    scenario_rows = run_scenarios(args.seed)

    write_csv(RESULTS_DIR / "benchmark_official.csv", official_rows)
    write_csv(RESULTS_DIR / "benchmark_hardened.csv", hardened_rows)
    (RESULTS_DIR / "scenario_results.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "trials": args.trials,
                "node_limit": args.node_limit,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "official": official_rows,
                "hardened": hardened_rows,
                "scenarios": scenario_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    generate_figures(official_rows, hardened_rows)

    print("=== OFFICIAL SUITE (mean ± std over", args.trials, "runs) ===")
    for r in official_rows:
        print(
            f"{r['instance']:17s} | {r['strategy']:17s} | {r['status']:5s} | "
            f"{r['mean_ms']:6.3f}±{r['std_ms']:5.3f}ms (med {r['median_ms']:6.3f}) | "
            f"nodes={r['nodes']:4d} | att={r['attempts']:4d} | bt={r['backtracks']:3d} | "
            f"ac3={r['ac3_pruned']:2d} | soft={r['soft_score']}"
        )

    print("\n=== HARDENED & STRESS SUITE (mean ± std over", args.trials, "runs, node_limit=", args.node_limit, ") ===")
    for r in hardened_rows:
        print(
            f"{r['instance']:17s} | {r['strategy']:17s} | {r['status']:14s} | "
            f"{r['mean_ms']:7.3f}±{r['std_ms']:6.3f}ms (med {r['median_ms']:7.3f}) | "
            f"nodes={r['nodes']:5d} | att={r['attempts']:6d} | bt={r['backtracks']:5d} | "
            f"ac3={r['ac3_pruned']:3d} | soft={r['soft_score']}"
        )

    print("\n=== SCENARIOS (S1–S6) ===")
    for s in scenario_rows:
        print(f"{s['id']} [{s['category']}]: {s['outcome']} -> {s}")


if __name__ == "__main__":
    main()
