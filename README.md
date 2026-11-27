# Project 07 — University Timetable Scheduler (Final Product)

- **Course:** Artificial Intelligence (Faculty of Information Technology, Hanoi University)
- **Group:** G13
- **GitHub Repository:** https://github.com/doduythai2410-spec/FIT3AIN-G13-P07-timetable-scheduler.git
- **Execution Mode:** 100% offline (no external CSP solver libraries, no hosted LLM APIs)

## 1. Environment Setup (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Launch the Interactive Application (Single Command)

```bash
streamlit run starter/app.py
```

## 3. Reproduce All Report Tables and Figures

The following command reproduces the 30-trial official benchmark (Table 3), the hardened/stress suite (Table 4), scenario evaluation results (Table 5), and all four 300-DPI black-and-white report figures under `results/`:

```bash
python scripts/run_experiment.py --seed 42 --trials 30
```

Generated artifacts in `results/`:
- `benchmark_official.csv`
- `benchmark_hardened.csv`
- `scenario_results.json`
- `fig1_architecture.png`
- `fig2_official_runtime_and_quality.png`
- `fig3_hardened_benchmark.png`
- `fig4_soft_optimization_breakdown.png`

## 4. Run Automated Tests

```bash
pytest tests/test_project_07_solver.py -v
```

*(Note: `tests/test_project_07_sanity.py` is the original starter-kit sanity script whose last two tests check for `NotImplementedError` on empty stubs. Use `tests/test_project_07_solver.py` to test the completed AI core and optimizer.)*

## 5. Directory Layout

```text
code/
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   ├── instance_feasible/
│   ├── instance_restricted_rooms/
│   ├── instance_extra_lecturer/
│   └── instance_unsat/
├── starter/
│   ├── loaders.py          # CSV/JSON schema validation, domain generation, independent validator
│   ├── student_core.py     # Graded AI core (AC-3, MRV+Degree, Incremental FC, Local Search)
│   └── app.py              # Interactive Streamlit timetable dashboard
├── scripts/
│   └── run_experiment.py   # Reproducible benchmark & figure generation script (--seed 42)
└── tests/
    ├── test_project_07_sanity.py
    └── test_project_07_solver.py

```
