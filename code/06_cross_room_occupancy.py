"""
Does the room-A model work in a different room? (Table 9)
Room A: UCI office in Belgium, 1-min readings. Room B: UCI room in India, 30-s readings, with PIR.
Shared sensors: light, CO2, temperature (room B's four nodes averaged).
Tests: within each room, A -> B and B -> A, raw and after a label-free rescaling.
Run: python3 06_cross_room_occupancy.py  -> ../results/occupancy/
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import RANDOM_STATE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(HERE, '..', 'results', 'occupancy')
os.makedirs(OUT, exist_ok=True)


def load_rooms():
    a_dir = os.path.join(DATA, 'occupancy+detection')
    a_train = pd.read_csv(os.path.join(a_dir, 'datatraining.txt'))
    a_test = pd.read_csv(os.path.join(a_dir, 'datatest2.txt'))
    for d in (a_train, a_test):
        d['occ'] = d.Occupancy
    b = pd.read_csv(os.path.join(DATA, 'room_occupancy_estimation', 'Occupancy_Estimation.csv'))
    b['occ'] = (b.Room_Occupancy_Count > 0).astype(int)   # 1+ people = occupied
    b['Light'] = b[['S1_Light', 'S2_Light', 'S3_Light', 'S4_Light']].mean(axis=1)
    b['Temperature'] = b[['S1_Temp', 'S2_Temp', 'S3_Temp', 'S4_Temp']].mean(axis=1)
    b['CO2'] = b['S5_CO2']
    b['PIR'] = b[['S6_PIR', 'S7_PIR']].max(axis=1)        # either PIR fired
    b['date'] = pd.to_datetime(b.Date)
    b_train, b_test = b[b.date < '2018-01-01'], b[b.date >= '2018-01-01']   # Dec -> Jan
    return a_train, a_test, b, b_train, b_test


def models():
    return {
        'LR': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
        'RF': RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_STATE),
        'XGB': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, eval_metric='logloss',
                             n_jobs=-1, random_state=RANDOM_STATE),
    }


def evaluate(y, p):
    yhat = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    # false vacancy = occupied but called empty (the risky mistake)
    return dict(auc=auc, accuracy=accuracy_score(y, yhat), false_vacancy=fn / max(1, fn + tp),
                missed_vacancy=fp / max(1, fp + tn), n=len(y), occupied=float(np.mean(y)))


def standardise(df, cols, ref):
    # rescale with the room's own mean and SD, no labels needed
    out = df.copy()
    for c in cols:
        sd = ref[c].std()
        out[c] = (df[c] - ref[c].mean()) / (sd if sd > 0 else 1.0)
    return out


def run(train, test, cols, setting, fs, rows):
    for name, m in models().items():
        m.fit(train[cols], train.occ)
        rows.append(dict(setting=setting, features=fs, model=name,
                         **evaluate(test.occ.values, m.predict_proba(test[cols])[:, 1])))


def main():
    a_train, a_test, b, b_train, b_test = load_rooms()
    a_all = pd.concat([a_train, a_test])
    shared = {'Light + CO2 + temperature': ['Light', 'CO2', 'Temperature'],
              'Light + CO2': ['Light', 'CO2'],
              'Light only': ['Light'],
              'CO2 only': ['CO2'],
              'CO2 + temperature (no light)': ['CO2', 'Temperature']}
    rows = []
    for fs, cols in shared.items():
        run(a_train, a_test, cols, 'within A (reference)', fs, rows)
        run(b_train, b_test, cols, 'within B (Dec -> Jan)', fs, rows)
        run(a_train, b, cols, 'A -> B, raw', fs, rows)
        run(b, a_test, cols, 'B -> A, raw', fs, rows)
        run(standardise(a_train, cols, a_all), standardise(b, cols, b), cols, 'A -> B, calibrated', fs, rows)
        run(standardise(b, cols, b), standardise(a_test, cols, a_all), cols, 'B -> A, calibrated', fs, rows)
    # PIR only exists in room B
    for fs, cols in {'PIR only': ['PIR'], 'CO2 + PIR (no light)': ['CO2', 'PIR'],
                     'Light + CO2 + PIR': ['Light', 'CO2', 'PIR']}.items():
        run(b_train, b_test, cols, 'within B (Dec -> Jan)', fs, rows)

    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(OUT, 'cross_room.csv'), index=False)
    pd.set_option('display.width', 200)
    print(res.pivot_table(index=['features', 'model'], columns='setting', values='auc', sort=False).round(3).to_string())
    print()
    print(res[res.model == 'LR'].pivot_table(index='features', columns='setting', values='false_vacancy', sort=False)
          .round(3).to_string())
    info = dict(room_A=dict(train=len(a_train), test=len(a_test)),
                room_B=dict(total=len(b), train_dec=len(b_train), test_jan=len(b_test),
                            occupied_share=float(b.occ.mean()), days=int(b.Date.nunique())))
    with open(os.path.join(OUT, 'cross_room_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info))


if __name__ == '__main__':
    main()
