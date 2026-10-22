# Data

The five datasets are public and are not redistributed here. Download them and place them in this folder:

```
data/
  hotel_bookings.csv
  KaggleV2-May-2016.csv
  occupancy+detection/
    datatraining.txt
    datatest.txt
    datatest2.txt
  room_occupancy_estimation/
    Occupancy_Estimation.csv
  robod/
    combined_Room1.csv ... combined_Room5.csv
```

| File | Dataset | Source |
|---|---|---|
| `hotel_bookings.csv` | Hotel booking demand (Antonio, Almeida and Nunes, 2019), 119,390 bookings | https://doi.org/10.1016/j.dib.2018.11.126 |
| `KaggleV2-May-2016.csv` | Medical appointment no-shows (Joni Hoppen and Aquarela Analytics), Vitória, Brazil, 110,527 appointments; licence CC BY-NC-SA 4.0 | https://www.kaggle.com/datasets/joniarroba/noshowappointments |
| `occupancy+detection/*.txt` | Occupancy detection (Candanedo and Feldheim, 2016), UCI Machine Learning Repository (room A) | https://doi.org/10.24432/C5X01N |
| `room_occupancy_estimation/Occupancy_Estimation.csv` | Room occupancy estimation (Singh and Chaudhari, 2018), UCI Machine Learning Repository (room B, cross-room validation) | https://doi.org/10.24432/C5P605 |
| `robod/combined_Room*.csv` | ROBOD (Tekler et al., 2022), five rooms in one building, CC BY 4.0 (building-level validation). Note: `07_building_occupancy.py` shifts the indoor environmental channels by 8 hours of clock time (the files skip weekends) to align them with the occupancy labels; see the script header | https://doi.org/10.6084/m9.figshare.19234530.v7 |

Run `python3 code/verify_data.py` to check that your files are identical (SHA-256) to the ones used in the paper.
