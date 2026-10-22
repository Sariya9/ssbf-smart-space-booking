#!/bin/sh
# Reproduces every table and figure in the paper. About 10 minutes on an 8-core laptop.
set -e
cd "$(dirname "$0")"
python3 code/verify_data.py
cd code
python3 01_noshow_model.py
python3 02_medical_noshow_model.py
python3 03_queue_simulation.py
python3 04_occupancy_model.py
python3 06_cross_room_occupancy.py
python3 07_building_occupancy.py
python3 07_building_occupancy.py --no-clock-correction
python3 08_room_adaptation.py
python3 09_sensor_fusion.py
python3 05_figures.py
echo "Done. Tables: results/   Figures: figures/"
