"""
Five rooms in one building, ROBOD (Table 10).
Lecture rooms, offices and a library in Singapore, 5-min data, real presence labels.
In the published files the indoor sensors are stamped 8 h early (light peaks at 2-6 am), so we
shift them back. Wi-Fi and energy meters are fine as they are. --no-clock-correction skips the fix.
Tests: within each room (70/30 by days), the other four rooms -> this room, and room A <-> ROBOD.
Run: python3 07_building_occupancy.py  -> ../results/occupancy/
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RANDOM_STATE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, '..', 'results', 'occupancy')
os.makedirs(OUT, exist_ok=True)

CORRECT_ENV_CLOCK = True
ENV = ['voc [ppb]', 'sound_pressure_level [dba]', 'indoor_relative_humidity [%]', 'air_temperature [Celsius]',
       'illuminance [lux]', 'pm2.5 [mu_g/m3]', 'indoor_co2 [ppm]']
ROOMS = {1: 'Lecture room 1', 2: 'Lecture room 2', 3: 'Office (admin)', 4: 'Office (research)', 5: 'Library'}
RENAME = {'illuminance [lux]': 'Light', 'indoor_co2 [ppm]': 'CO2', 'air_temperature [Celsius]': 'Temperature',
          'indoor_relative_humidity [%]': 'Humidity', 'wifi_connected_devices [number]': 'WiFi',
          'lighting_energy [kWh]': 'LightingEnergy', 'plug_load_energy [kWh]': 'PlugEnergy'}
FEATURES = {
    'Light + CO2 + temperature': ['Light', 'CO2', 'Temperature'],
    'CO2 + temperature (no light)': ['CO2', 'Temperature'],
    'Wi-Fi devices only': ['WiFi'],
    'Environmental + Wi-Fi': ['Light', 'CO2', 'Temperature', 'Humidity', 'WiFi'],
}
SHARED = ['Light + CO2 + temperature', 'CO2 + temperature (no light)']      # room A has these too


def load_robod():
    rooms = {}
    for r in ROOMS:
        d = pd.read_csv(os.path.join(DATA, 'robod', f'combined_Room{r}.csv'))
        if CORRECT_ENV_CLOCK:
            # shift by clock time, not rows: the files skip weekends
            ts = pd.to_datetime(d.timestamp.str[:16])
            d[ENV] = d[ENV].set_index(ts).reindex(ts - pd.Timedelta(hours=8)).values
        d = d.rename(columns=RENAME)
        d['occ'] = d['occupant_presence [binary]']
        d['ts'] = pd.to_datetime(d.timestamp.str[:16])
        d = d.dropna(subset=['occ', 'Light', 'CO2', 'Temperature', 'Humidity', 'WiFi']).reset_index(drop=True)
        d['room'] = r
        rooms[r] = d
    return rooms


def load_uci():
    a = os.path.join(DATA, 'occupancy+detection')
    tr, te = pd.read_csv(os.path.join(a, 'datatraining.txt')), pd.read_csv(os.path.join(a, 'datatest2.txt'))
    for d in (tr, te):
        d['occ'] = d.Occupancy
    return tr, te


def models():
    return {'LR': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
            'RF': RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced',
                                         n_jobs=1, random_state=RANDOM_STATE)}   # 1 thread so reruns match exactly


def evaluate(y, p):
    tn, fp, fn, tp = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y, p), false_vacancy=fn / max(1, fn + tp), missed_vacancy=fp / max(1, fp + tn),
                n=len(y), occupied=float(np.mean(y)))


def calibrate(df, cols):
    out = df.copy()
    for c in cols:
        sd = df[c].std()
        out[c] = (df[c] - df[c].mean()) / (sd if sd > 0 else 1.0)
    return out


def fit_eval(train, test, cols, **tag):
    rows = []
    for name, m in models().items():
        m.fit(train[cols], train.occ)
        rows.append(dict(model=name, **tag, **evaluate(test.occ.values, m.predict_proba(test[cols])[:, 1])))
    return rows


def main():
    global CORRECT_ENV_CLOCK
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-clock-correction', action='store_true')
    CORRECT_ENV_CLOCK = not ap.parse_args().no_clock_correction
    sfx = '' if CORRECT_ENV_CLOCK else '_uncorrected'
    rooms = load_robod()
    uci_tr, uci_te = load_uci()
    rows = []
    for fs, cols in FEATURES.items():
        for r, d in rooms.items():
            days = np.sort(d.ts.dt.normalize().unique())
            cut = days[int(len(days) * 0.7)]
            rows += fit_eval(d[d.ts < cut], d[d.ts >= cut], cols, setting='within room', features=fs, room=ROOMS[r])
            others = pd.concat([rooms[o] for o in rooms if o != r])
            rows += fit_eval(others, d, cols, setting='leave-one-room-out, raw', features=fs, room=ROOMS[r])
            cal_others = pd.concat([calibrate(rooms[o], cols) for o in rooms if o != r])
            rows += fit_eval(cal_others, calibrate(d, cols), cols, setting='leave-one-room-out, calibrated',
                             features=fs, room=ROOMS[r])
            if fs in SHARED:
                rows += fit_eval(uci_tr, d, cols, setting='UCI office -> room, raw', features=fs, room=ROOMS[r])
                uci_all = pd.concat([uci_tr, uci_te])
                cal_tr = calibrate(uci_tr, cols)
                for c in cols:   # rescale with all room A readings
                    cal_tr[c] = (uci_tr[c] - uci_all[c].mean()) / uci_all[c].std()
                rows += fit_eval(cal_tr, calibrate(d, cols), cols, setting='UCI office -> room, calibrated',
                                 features=fs, room=ROOMS[r])
        if fs in SHARED:
            robod_all = pd.concat(rooms.values())
            rows += fit_eval(robod_all, uci_te, cols, setting='ROBOD building -> UCI office, raw', features=fs,
                             room='UCI office')
    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(OUT, f'building{sfx}.csv'), index=False)

    lr = res[res.model == 'LR']
    summ = lr.groupby(['features', 'setting'], sort=False).agg(
        auc_median=('auc', 'median'), auc_min=('auc', 'min'), auc_max=('auc', 'max'),
        fv_median=('false_vacancy', 'median'), fv_max=('false_vacancy', 'max'),
        mv_median=('missed_vacancy', 'median')).round(3)
    summ.to_csv(os.path.join(OUT, f'building_summary{sfx}.csv'))
    pd.set_option('display.width', 220)
    print(summ.to_string())
    print()
    print(lr.pivot_table(index=['features', 'room'], columns='setting', values='auc', sort=False).round(3).to_string())
    info = {ROOMS[r]: dict(n=len(d), occupied=round(float(d.occ.mean()), 3), days=int(d.ts.dt.date.nunique()))
            for r, d in rooms.items()}
    info['env_clock_corrected'] = CORRECT_ENV_CLOCK
    with open(os.path.join(OUT, f'building_info{sfx}.json'), 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    main()
