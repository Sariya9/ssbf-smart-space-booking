# Smart Space Booking Framework (SSBF): code and data for the revised manuscript

Code accompanying *An Explainable AI Framework for Smart Space Management Integrating IoT Sensing, Machine Learning, and Fair Resource Allocation* (Lura, Sammo, Al-Quraishi, Bevinakoppa, Nowrozy), submitted to *Discover Artificial Intelligence*.

Every table and figure in the revised manuscript is produced by the scripts below. All random seeds are fixed, so a run on the same data reproduces the reported numbers exactly.

## Quick start

```sh
pip install -r requirements.txt
# put the five public datasets in data/ (see data/README.md)
sh run_all.sh          # about 7-10 minutes on an 8-core laptop
```

`run_all.sh` first checks the datasets against SHA-256 checksums, then runs the analysis scripts in order.

## What each script produces

| Script | Manuscript | Output |
|---|---|---|
| `code/01_noshow_model.py` | Tables 2, 3, 5 and 6 (hotel parts) | `results/noshow/` |
| `code/02_medical_noshow_model.py` | Tables 3, 5 and 6 (medical parts) | `results/medical/` |
| `code/03_queue_simulation.py` | Tables 4, 13, 14, 15 | `results/simulation/` |
| `code/04_occupancy_model.py` | Tables 7, 8 | `results/occupancy/` |
| `code/06_cross_room_occupancy.py` | Table 9 (cross-room validation) | `results/occupancy/cross_room*` |
| `code/07_building_occupancy.py` | Table 10 (building-level validation; `--no-clock-correction` for the uncorrected run) | `results/occupancy/building*` |
| `code/08_room_adaptation.py` | Fig. 6 and part of Table 11 (commissioning a new room; reuses 03 and 07) | `results/occupancy/adaptation*`, `results/simulation/adaptation_scenarios.csv`, `figures/r1_adaptation.png` |
| `code/09_sensor_fusion.py` | Table 12 and part of Table 11 (sensor fusion and its simulated release outcomes; reuses 03, 06, 07) | `results/occupancy/fusion*`, `results/simulation/fusion_scenarios.csv` |
| `code/05_figures.py` | Fig. 2 (`fig12_sensor_timeseries.png`) and Figs. 3–5 and 7–11 (`r1_*.png`), SHAP values | `figures/`, `results/shap_*` |
| `code/common.py` | Model definitions shared by all prediction scripts | |

Table and figure numbers refer to the revised manuscript. Fig. 1 (`figures/fig1_architecture.pdf`) is a diagram drawn in LaTeX, not produced by the code.

## Key design choices

- **Target.** Hotel no-shows are `reservation_status == "No-Show"` versus `"Check-Out"`; cancelled bookings are excluded, because an advance cancellation releases capacity and is not a ghost booking.
- **Features.** Only information available when a booking is created. Excluded variables and the reasons are listed in each script (`EXCLUDED`).
- **Validation.** Stratified (hotel) or patient-grouped (medical) 5-fold CV, plus temporal hold-outs and cross-hotel transfer.
- **Simulation.** All parameters are in `DEFAULTS` in `03_queue_simulation.py`. No-show rates are specified independently of the ML models. The statistical unit is the replication (the same 30 seeds for the main experiment and every sensitivity setting), with common random numbers across conditions. Twelve conditions are compared, including extended redistribution and the late-arrival safeguards (a "running late" button and re-housing of displaced users); the safeguard conditions use their own random stream, so adding them leaves every other condition unchanged.

## Software

Python 3.13 with the package versions in `requirements.txt`.

## Data

See `data/README.md`. The datasets are public and are not redistributed in this repository.
