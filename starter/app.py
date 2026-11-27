"""Interactive Streamlit application for Project 07 (University Timetable Scheduler).

Integrates the CSP AI core (`student_core.py`) with dynamic dataset loading,
what-if scenario controls (closing rooms, blocking lecturer slots, adding
live daily teaching limits), interactive color-coded weekly calendar grids,
independent hard-constraint verification, soft-constraint quality analytics,
and multi-strategy benchmarking. Runs 100% offline.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from html import escape
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile

import pandas as pd
import streamlit as st

import loaders
import student_core

PALETTE = [
    ("#DBEAFE", "#1E40AF"),
    ("#DCFCE7", "#166534"),
    ("#FEF3C7", "#92400E"),
    ("#FCE7F3", "#9D174D"),
    ("#E0E7FF", "#3730A3"),
    ("#FFEDD5", "#9A3412"),
    ("#CCFBF1", "#115E59"),
    ("#F3E8FF", "#6B21A8"),
]

STRATEGY_LABELS = {
    "naive": "Naive Backtracking (Baseline)",
    "midterm_heuristic": "MRV + Copy-on-Branch FC (Midterm)",
    "heuristic": "AC-3 + MRV/Degree + Inc-FC (Final Fast)",
    "optimized": "Final Core + Soft-Constraint Optimizer (Recommended)",
}


def get_days_periods(instance: dict) -> tuple[list[str], list[int]]:
    """Preserve timeslot-file day order and sort class periods numerically."""
    days = list(dict.fromkeys(info["day"] for info in instance["slot_info"].values()))
    periods = sorted({int(info["period"]) for info in instance["slot_info"].values()})
    return days, periods


def timetable_matrix(instance: dict, assignment: dict | None) -> pd.DataFrame:
    """One row per period, one column per day, listing every scheduled course."""
    days, periods = get_days_periods(instance)
    grid = pd.DataFrame("", index=[f"Period {p}" for p in periods], columns=days)
    if not assignment:
        return grid
    labels = defaultdict(list)
    for cid, (lecturer, room, slot) in assignment.items():
        info = instance["slot_info"][slot]
        labels[(f"Period {info['period']}", info["day"])].append(
            f"{cid} ({lecturer}, {room})"
        )
    for (period, day), items in labels.items():
        grid.at[period, day] = "  |  ".join(sorted(items))
    return grid


def calendar_html(
    instance: dict,
    assignment: dict,
    color_by: str = "Lecturer",
    selected: set[str] | None = None,
) -> str:
    """Render escaped HTML with color-coded course cards inside each day/period cell."""
    if color_by not in {"Lecturer", "Room", "Cohort"}:
        raise ValueError("color_by must be Lecturer, Room or Cohort")
    selected = selected or set()
    days, periods = get_days_periods(instance)
    courses = {c["course_id"]: c for c in instance["courses"]}
    grouped = defaultdict(list)
    for cid, (lecturer, room, slot) in assignment.items():
        info = instance["slot_info"][slot]
        cohort = courses[cid]["cohort"]
        group = {"Lecturer": lecturer, "Room": room, "Cohort": cohort}[color_by]
        if selected and group not in selected:
            continue
        grouped[(info["day"], int(info["period"]))].append(
            (cid, courses[cid]["course_name"], lecturer, room, cohort, group)
        )
    html = ['<div class="tt-scroll"><table class="tt-table"><thead><tr><th>TIME</th>']
    html += [f"<th>{escape(day)}</th>" for day in days]
    html.append("</tr></thead><tbody>")
    for period in periods:
        slots = [
            info
            for info in instance["slot_info"].values()
            if int(info["period"]) == period
        ]
        time_lbl = f"{slots[0]['start_time']}–{slots[0]['end_time']}" if slots else ""
        html.append(f"<tr><th>Period {period}<small>{escape(time_lbl)}</small></th>")
        for day in days:
            html.append("<td>")
            courses_here = sorted(grouped.get((day, period), []))
            if not courses_here:
                html.append('<span class="tt-empty">—</span>')
            for cid, name, lecturer, room, cohort, group in courses_here:
                idx = (
                    int.from_bytes(sha256(group.encode("utf-8")).digest()[:2], "big")
                    % len(PALETTE)
                )
                bg, fg = PALETTE[idx]
                html.append(
                    f'<div class="tt-course" style="background:{bg};color:{fg};border-left-color:{fg}">'
                    f"<strong>{escape(cid)} — {escape(name)}</strong>"
                    f"<small>{escape(lecturer)} · {escape(room)} · {escape(cohort)}</small>"
                    "</div>"
                )
            html.append("</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)


CALENDAR_CSS = """
<style>
.tt-scroll{overflow-x:auto;border:1px solid #CBD5E1;border-radius:12px;margin:10px 0}
.tt-table{width:100%;min-width:870px;table-layout:fixed;border-collapse:collapse}
.tt-table th{padding:12px;text-align:left;background:rgba(148,163,184,.16);font-size:12px}
.tt-table th:first-child{width:135px}
.tt-table th small{display:block;font-size:11px;font-weight:normal;margin-top:5px;opacity:.7}
.tt-table td{padding:8px;vertical-align:top;border-top:1px solid #CBD5E1;border-left:1px solid #CBD5E1;height:118px}
.tt-course{border-left:4px solid;border-radius:7px;padding:7px 9px;margin-bottom:8px;overflow-wrap:anywhere}
.tt-course strong{display:block;font-size:12px;margin-bottom:5px}
.tt-course small{font-size:11px;display:block}
.tt-empty{opacity:.35}
</style>
"""

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INSTANCE_NAMES = ("feasible", "restricted_rooms", "extra_lecturer", "unsat")
REQUIRED_FILES = {
    "courses.csv",
    "lecturers.csv",
    "rooms.csv",
    "timeslots.csv",
    "constraints.json",
}


def read_custom_instance(upload) -> dict:
    """Accept exactly five standard CSV/JSON files inside a ZIP archive (<= 2 MB)."""
    data = upload.getvalue()
    if len(data) > 2_000_000:
        raise ValueError("Upload must be at most 2 MB")
    with ZipFile(BytesIO(data)) as archive:
        entries = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir() and not entry.filename.startswith("__MACOSX/")
        ]
        basenames = [Path(entry.filename).name for entry in entries]
        if (
            len(entries) != 5
            or set(basenames) != REQUIRED_FILES
            or len(set(basenames)) != 5
        ):
            raise ValueError(
                "ZIP must contain exactly the five files: courses.csv, lecturers.csv, "
                "rooms.csv, timeslots.csv, constraints.json"
            )
        if sum(entry.file_size for entry in entries) > 2_000_000:
            raise ValueError("Uncompressed ZIP exceeds 2 MB")
        with TemporaryDirectory(prefix="timetable_") as temp:
            for entry in entries:
                (Path(temp) / Path(entry.filename).name).write_bytes(
                    archive.read(entry)
                )
            return loaders.load_instance(Path(temp))


def load_instance(name: str, upload) -> dict | None:
    if name == "Custom ZIP":
        return read_custom_instance(upload) if upload is not None else None
    return loaders.load_instance(DATA_DIR / f"instance_{name}")


def display_metrics(record: dict | None) -> None:
    """Display primary runtime, search nodes, validity, and soft-score metrics."""
    if record is None:
        runtime, nodes, validity, soft_val = "—", "—", "Not evaluated", "—"
    else:
        stats = record["stats"]
        runtime = f"{stats['runtime_ms']:.3f}"
        nodes = stats["nodes"]
        if record["solution"] is None:
            validity = "N/A (UNSAT)"
            soft_val = "N/A"
        else:
            validity = "Valid" if record["valid"] else "Invalid"
            fs = stats.get("final_soft_score")
            soft_val = f"{fs:.2f} / 100" if fs is not None else "—"

    cols = st.columns(4)
    cols[0].metric("Runtime (ms)", runtime)
    cols[1].metric("Nodes Explored", nodes)
    cols[2].metric("Solution Validity", validity)
    cols[3].metric("Soft Quality Score", soft_val)

    if record is not None:
        stats = record["stats"]
        if record["solution"] is None:
            st.caption(
                "UNSAT: no complete assignment exists, so solution validity is not applicable."
            )
        with st.expander("Detailed search & optimization statistics"):
            e1 = st.columns(4)
            e1[0].metric("Search Status", stats["status"])
            e1[1].metric("Candidate Attempts", stats["attempts"])
            e1[2].metric("Backtracks", stats["backtracks"])
            e1[3].metric("Values Pruned (AC-3 / Total)", f"{stats.get('ac3_pruned', 0)} / {stats.get('pruned_values', 0)}")
            if stats.get("strategy") == "optimized" and record["solution"] is not None:
                e2 = st.columns(4)
                e2[0].metric("CSP Search Time (ms)", f"{stats.get('search_ms', 0.0):.3f}")
                e2[1].metric("Local Search Time (ms)", f"{stats.get('opt_ms', 0.0):.3f}")
                e2[2].metric("Initial Soft Score", f"{stats.get('initial_soft_score', 0.0):.2f}")
                gain = (stats.get("final_soft_score") or 0.0) - (stats.get("initial_soft_score") or 0.0)
                e2[3].metric("Soft Score Gain", f"+{gain:.2f} ({stats.get('opt_steps', 0)} steps)")


def main() -> None:
    st.set_page_config(
        page_title="University Timetable Scheduler", page_icon="📅", layout="wide"
    )
    st.markdown(CALENDAR_CSS, unsafe_allow_html=True)
    st.title("University Timetable Scheduler")
    st.caption(
        "Project 07 · Constraint Satisfaction Problem (CSP) · "
        "AC-3 + MRV/Degree + Incremental Forward Checking + Soft-Constraint Optimizer"
    )

    with st.sidebar:
        st.header("Controls")
        instance_name = st.selectbox("Dataset", list(INSTANCE_NAMES) + ["Custom ZIP"])
        upload = (
            st.file_uploader("Instance ZIP (five data files)", type=["zip"])
            if instance_name == "Custom ZIP"
            else None
        )
        strategy = st.selectbox(
            "Solver strategy",
            ["optimized", "heuristic", "midterm_heuristic", "naive"],
            format_func=lambda v: STRATEGY_LABELS[v],
        )
        st.markdown("**Live Scenario Controls (optional)**")

    try:
        original = load_instance(instance_name, upload)
    except (
        loaders.SchemaError,
        ValueError,
        OSError,
        KeyError,
        UnicodeDecodeError,
        BadZipFile,
    ) as exc:
        st.error(f"Data loading error: {exc}")
        st.stop()
    if original is None:
        st.info("Upload a ZIP containing the five CSV/JSON input files to begin.")
        st.stop()

    with st.sidebar:
        close_room = st.selectbox(
            "Temporarily close a room", ["None"] + list(original["rooms"])
        )
        restrict_lecturer = st.selectbox(
            "Block a lecturer timeslot", ["None"] + list(original["lecturers"])
        )
        block_slot = st.selectbox(
            "Timeslot to remove from lecturer",
            original["all_slots"],
            disabled=restrict_lecturer == "None",
        )
        limit_lecturer = st.selectbox(
            "Add daily limit rule for lecturer", ["None"] + list(original["lecturers"])
        )
        daily_cap = st.number_input(
            "Max courses per day",
            min_value=1,
            max_value=4,
            value=1,
            step=1,
            disabled=limit_lecturer == "None",
        )
        generate = st.button(
            "Generate timetable", type="primary", use_container_width=True
        )
        clear = st.button("Clear result", use_container_width=True)
        st.caption(
            "Original CSV/JSON files on disk are never modified by scenario controls."
        )

    instance = deepcopy(original)
    if close_room != "None":
        del instance["rooms"][close_room]
    if restrict_lecturer != "None":
        instance["lecturers"][restrict_lecturer]["available_slots"].discard(block_slot)
    if limit_lecturer != "None":
        instance["constraints"].setdefault("extra_constraints", []).append(
            {
                "type": "lecturer_daily_limit",
                "lecturer_id": limit_lecturer,
                "max_per_day": int(daily_cap),
                "reason": "Interactive UI scenario rule",
            }
        )

    key_material = (
        f"{instance_name}|{strategy}|{close_room}|{restrict_lecturer}|"
        f"{block_slot}|{limit_lecturer}|{daily_cap}"
    ).encode("utf-8")
    if upload is not None:
        key_material += upload.getvalue()
    result_key = sha256(key_material).hexdigest()

    cols = st.columns(4)
    for col, label, value in zip(
        cols,
        ["Courses", "Lecturers", "Rooms", "Timeslots"],
        [
            len(instance["courses"]),
            len(instance["lecturers"]),
            len(instance["rooms"]),
            len(instance["all_slots"]),
        ],
    ):
        col.metric(label, value)
    st.caption(instance["constraints"].get("description", ""))

    if clear:
        st.session_state.pop("timetable_result", None)
        st.session_state.pop("strategy_comparison", None)

    if generate:
        try:
            with st.spinner("Searching for a valid timetable..."):
                solution = student_core.solve_csp(instance, strategy=strategy)
                stats = student_core.SEARCH_STATS.copy()
                valid, errors = (
                    loaders.validate_assignment(instance, solution)
                    if solution is not None
                    else (False, [])
                )
                st.session_state["timetable_result"] = {
                    "key": result_key,
                    "solution": solution,
                    "stats": stats,
                    "valid": valid,
                    "errors": errors,
                }
        except Exception as exc:
            st.session_state.pop("timetable_result", None)
            st.error(f"Solver error: {type(exc).__name__}: {exc}")

    record = st.session_state.get("timetable_result")
    if record is not None and record["key"] != result_key:
        record = None
        st.info("Inputs changed. Click Generate timetable to solve the new scenario.")

    if record is not None:
        if record["solution"] is None:
            st.warning("UNSAT: No timetable satisfies all hard constraints.")
            for reason in student_core.diagnose_unsat(instance):
                st.write("• " + reason)
        elif record["valid"]:
            st.success("SAT: A complete, hard-constraint-valid timetable was found.")
        else:
            st.error("Invalid assignment: " + "; ".join(record["errors"]))
    else:
        st.info("Choose a dataset and strategy, then click Generate timetable.")

    solution = record["solution"] if record and record["valid"] else None
    calendar_tab, assignments_tab, validation_tab, data_tab = st.tabs(
        [
            "📅 Timetable",
            "📋 Assignments & Quality",
            "✅ Constraint Checker",
            "📊 Experiments & Data",
        ]
    )

    with calendar_tab:
        st.subheader("Solver Performance & Solution Validity")
        display_metrics(record)
        st.subheader("Weekly timetable — courses inside each cell")
        if solution:
            choice_col, filter_col = st.columns([1, 2])
            with choice_col:
                color_by = st.selectbox("Color-code by", ["Lecturer", "Room", "Cohort"])
            courses_by_id = {c["course_id"]: c for c in instance["courses"]}
            categories = sorted(
                {
                    {
                        "Lecturer": ch[0],
                        "Room": ch[1],
                        "Cohort": courses_by_id[cid]["cohort"],
                    }[color_by]
                    for cid, ch in solution.items()
                }
            )
            with filter_col:
                selected = st.multiselect(
                    f"Only show these {color_by.lower()} values (empty = all)",
                    categories,
                )
            st.markdown(
                calendar_html(instance, solution, color_by, set(selected)),
                unsafe_allow_html=True,
            )
            with st.expander("Compact DataFrame grid"):
                st.dataframe(
                    timetable_matrix(instance, solution), use_container_width=True
                )
            st.caption(
                "Multiple courses can appear in one day/period cell if they use "
                "different lecturers, rooms, and cohorts."
            )
        else:
            st.dataframe(timetable_matrix(instance, None), use_container_width=True)
            st.caption("This table will be populated after a valid timetable is generated.")

    with assignments_tab:
        st.subheader("Assigned courses & soft-constraint analytics")
        if solution:
            rows = loaders.assignment_rows(instance, solution)
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            try:
                bd = student_core.soft_constraint_breakdown(instance, solution)
                q1, q2, q3 = st.columns(3)
                q1.metric("Combined Soft Score", f"{bd['total_score']:.2f} / 100")
                q2.metric("Cohort Daily Spread", f"{bd['cohort_daily_spread_pct']:.2f}%")
                q3.metric(
                    "Lecturer Load Balance", f"{bd['lecturer_load_balance_pct']:.2f}%"
                )

                ch1, ch2 = st.columns(2)
                with ch1:
                    st.markdown("**Courses per Lecturer**")
                    load_df = pd.DataFrame(
                        [
                            {"Lecturer": lid, "Courses": cnt}
                            for lid, cnt in sorted(bd["lecturer_loads"].items())
                        ]
                    ).set_index("Lecturer")
                    st.bar_chart(load_df)
                with ch2:
                    st.markdown("**Distinct Teaching Days per Cohort**")
                    coh_df = pd.DataFrame(
                        [
                            {
                                "Cohort": coh,
                                "Courses": info["courses"],
                                "Distinct Days": info["distinct_days"],
                                "Days": ", ".join(info["days"]),
                            }
                            for coh, info in sorted(bd["cohort_details"].items())
                        ]
                    )
                    st.dataframe(coh_df, hide_index=True, use_container_width=True)
            except ValueError as exc:
                st.caption(f"Soft score unavailable: {exc}")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download timetable CSV",
                    loaders.assignment_csv(instance, solution),
                    file_name=f"timetable_{instance_name}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Download assignment JSON",
                    json.dumps(solution, indent=2),
                    file_name=f"assignment_{instance_name}.json",
                    mime="application/json",
                    use_container_width=True,
                )
        else:
            st.info("Generate a SAT solution to view quality metrics or export assignments.")

    with validation_tab:
        st.subheader("Independent hard-constraint verification")
        if solution:
            report = loaders.constraint_report(instance, solution)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Constraint": row["constraint"],
                            "Status": row["status"],
                            "Violations": row["violations"],
                        }
                        for row in report
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.markdown(
                "**What-if validator:** Edit one course assignment to verify violation detection."
            )
            course_id = st.selectbox("Course to edit", list(solution))
            before = solution[course_id]
            e1, e2, e3 = st.columns(3)
            with e1:
                lecturers = list(instance["lecturers"])
                new_lecturer = st.selectbox(
                    "Lecturer", lecturers, index=lecturers.index(before[0])
                )
            with e2:
                rooms = list(instance["rooms"])
                new_room = st.selectbox("Room", rooms, index=rooms.index(before[1]))
            with e3:
                slots = instance["all_slots"]
                new_slot = st.selectbox("Timeslot", slots, index=slots.index(before[2]))
            changed = dict(solution)
            changed[course_id] = (new_lecturer, new_room, new_slot)
            is_ok, errs = loaders.validate_assignment(instance, changed)
            if is_ok:
                st.success("What-if assignment satisfies all hard constraints.")
            else:
                st.error(f"{len(errs)} violation(s) detected:")
                for err in errs:
                    st.write("• " + err)
        elif record and record["solution"] is None:
            st.info(
                "No assignment exists to check (UNSAT). All possibilities were exhausted."
            )
        else:
            st.info("Generate a timetable to inspect its constraint verification report.")

    with data_tab:
        st.subheader("Explore the CSP input & compare strategies")
        with st.expander("Courses", expanded=False):
            st.dataframe(
                pd.DataFrame(instance["courses"]),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("Lecturers"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "lecturer_id": lid,
                            "name": info["name"],
                            "available_slots": len(info["available_slots"]),
                        }
                        for lid, info in instance["lecturers"].items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        with st.expander("Rooms"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "room_id": rid,
                            "type": info["room_type"],
                            "capacity": info["capacity"],
                        }
                        for rid, info in instance["rooms"].items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )
        with st.expander("Constraints"):
            st.json(instance["constraints"])
        with st.expander("Initial domain sizes"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "course_id": course["course_id"],
                            "possible_assignments": len(
                                loaders.candidate_domain(course, instance)
                            ),
                        }
                        for course in instance["courses"]
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("**Compare all four strategies (single run)**")
        st.caption(
            "For formal multi-trial statistics, run `python scripts/run_experiment.py --seed 42`."
        )
        if st.button("Compare all strategies"):
            comparison = []
            for method in ("naive", "midterm_heuristic", "heuristic", "optimized"):
                found = student_core.solve_csp(instance, strategy=method)
                measured = student_core.SEARCH_STATS.copy()
                is_valid = (
                    loaders.validate_assignment(instance, found)[0]
                    if found is not None
                    else False
                )
                comparison.append(
                    {
                        "Strategy": STRATEGY_LABELS[method],
                        "Status": measured["status"],
                        "Runtime (ms)": round(float(measured["runtime_ms"]), 3),
                        "Nodes": measured["nodes"],
                        "Attempts": measured["attempts"],
                        "Backtracks": measured["backtracks"],
                        "Soft Score": (
                            measured["final_soft_score"]
                            if found is not None
                            else "N/A"
                        ),
                        "Valid": is_valid if found is not None else "UNSAT",
                    }
                )
            st.session_state["strategy_comparison"] = (result_key, comparison)

        comparison = st.session_state.get("strategy_comparison")
        if comparison is not None and comparison[0] == result_key:
            st.dataframe(
                pd.DataFrame(comparison[1]), use_container_width=True, hide_index=True
            )


if __name__ == "__main__":
    main()
