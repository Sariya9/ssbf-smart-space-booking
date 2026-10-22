"""
Occupancy detection in UCI room A (Tables 7 and 8).
UCI files: datatraining.txt = train (8,143), datatest.txt = Test Set 1 (2,665),
datatest2.txt = Test Set 2 (9,752).
Run: python3 04_occupancy_model.py  -> ../results/occupancy/
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import RANDOM_STATE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'occupancy+detection')
OUT = os.path.join(HERE, '..', 'results', 'occupancy')
os.makedirs(OUT, exist_ok=True)

SENSORS = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']


def models():
    return {
        'LR': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
        'RF': RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced',
                                     n_jobs=-1, random_state=RANDOM_STATE),
        'XGB': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, eval_metric='logloss',
                             n_jobs=-1, random_state=RANDOM_STATE),
    }


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= 0.5).astype(int)
    # occupied = 1, so fnr is the false vacancy rate
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return dict(auc=roc_auc_score(y, p), accuracy=accuracy_score(y, yhat), f1=f1_score(y, yhat),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), fpr=fp / (fp + tn), fnr=fn / (fn + tp), n=len(y))


def main():
    read = lambda f: pd.read_csv(os.path.join(DATA, f))
    train, test1, test2 = read('datatraining.txt'), read('datatest.txt'), read('datatest2.txt')
    info = {name: dict(n=len(df), start=str(df.date.min()), end=str(df.date.max()),
                       occupied=float(df.Occupancy.mean()))
            for name, df in [('train', train), ('test1', test1), ('test2', test2)]}
    print(json.dumps(info, indent=1))

    rows = []
    feature_sets = {'all five sensors': SENSORS, 'Light + CO2': ['Light', 'CO2'], 'CO2 only': ['CO2'],
                    'without Light': [s for s in SENSORS if s != 'Light']}
    for fs_name, cols in feature_sets.items():
        for name, m in models().items():
            m.fit(train[cols], train.Occupancy)
            for tname, tdf in [('Test Set 1 (2,665)', test1), ('Test Set 2 (9,752)', test2)]:
                rows.append(dict(features=fs_name, model=name, test=tname, **evaluate(m, tdf[cols], tdf.Occupancy)))
    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(OUT, 'occupancy_results.csv'), index=False)
    print(res[['features', 'model', 'test', 'auc', 'accuracy', 'f1', 'fpr', 'fnr', 'tn', 'fp', 'fn', 'tp']]
          .to_string(index=False))

    # what if the light sensor dies? replace it with its training mean
    rows = []
    for name, m in models().items():
        m.fit(train[SENSORS], train.Occupancy)
        broken = test2[SENSORS].copy()
        broken['Light'] = train.Light.mean()
        rows.append(dict(model=name, scenario='Light channel failed', **evaluate(m, broken, test2.Occupancy)))
    fail = pd.DataFrame(rows).round(4)
    fail.to_csv(os.path.join(OUT, 'sensor_failure.csv'), index=False)
    print('\nLight channel failure on Test Set 2\n', fail[['model', 'auc', 'accuracy', 'fpr', 'fnr']].to_string(index=False))
    with open(os.path.join(OUT, 'splits.json'), 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    main()
