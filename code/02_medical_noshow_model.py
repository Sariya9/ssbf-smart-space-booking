"""
Medical appointment no-show model (Tables 5 and 6, and the simulation's ML settings).
No-show = booked but the patient never came, same as a ghost booking.
Features are what we know when the appointment is booked (ScheduledDay).
Run: python3 02_medical_noshow_model.py  -> ../results/medical/
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import RANDOM_STATE, make_models, recall_at_fpr, scores

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'KaggleV2-May-2016.csv')
OUT = os.path.join(HERE, '..', 'results', 'medical')
os.makedirs(OUT, exist_ok=True)

NUMERIC = ['lead_days', 'age', 'sched_hour', 'scholarship', 'hypertension', 'diabetes', 'alcoholism',
           'handicap', 'prior_appts', 'prior_noshows', 'prior_noshow_rate', 'is_female']
CATEGORICAL = ['sched_weekday', 'appt_weekday']

EXCLUDED = {
    'SMS_received': 'reminder sent after booking (only for longer lead times), not known at creation',
    'Neighbourhood': 'location identifier (81 levels); excluded so features stay transferable',
    'PatientId / AppointmentID': 'identifiers (PatientId used only for grouping and history)',
}


def load():
    d = pd.read_csv(DATA)
    d['sched'] = pd.to_datetime(d.ScheduledDay).dt.tz_localize(None)
    d['appt'] = pd.to_datetime(d.AppointmentDay).dt.tz_localize(None)
    d['lead_days'] = (d.appt.dt.normalize() - d.sched.dt.normalize()).dt.days
    d = d[(d.lead_days >= 0) & (d.Age >= 0)].copy()      # drop 6 bad rows
    d['y'] = (d['No-show'] == 'Yes').astype(int)
    d['age'] = d.Age
    d['sched_hour'] = d.sched.dt.hour
    d['sched_weekday'] = d.sched.dt.day_name()
    d['appt_weekday'] = d.appt.dt.day_name()
    d['is_female'] = (d.Gender == 'F').astype(int)
    d = d.rename(columns={'Scholarship': 'scholarship', 'Hipertension': 'hypertension', 'Diabetes': 'diabetes',
                          'Alcoholism': 'alcoholism', 'Handcap': 'handicap'})

    # past attendance, counting only appointments already over when this one was booked
    d = d.sort_values('appt').reset_index(drop=True)
    prior_appts = np.zeros(len(d))
    prior_ns = np.zeros(len(d))
    for _, g in d.groupby('PatientId'):
        if len(g) == 1:
            continue
        days = g.appt.dt.normalize().values
        ys = g.y.values
        sched_days = g.sched.dt.normalize().values
        for idx, sd in zip(g.index, sched_days):
            done = days < sd
            prior_appts[idx] = done.sum()
            prior_ns[idx] = ys[done].sum()
    d['prior_appts'] = prior_appts
    d['prior_noshows'] = prior_ns
    d['prior_noshow_rate'] = np.where(prior_appts > 0, prior_ns / np.maximum(prior_appts, 1), -1)
    return d


def main():
    t0 = time.time()
    d = load()
    print(f"Medical task: {len(d):,} appointments, {d.y.sum():,} no-shows ({d.y.mean()*100:.2f}%), "
          f"{d.PatientId.nunique():,} patients, {int((d.prior_appts > 0).sum()):,} with resolved history")

    X, y, groups = d[NUMERIC + CATEGORICAL], d.y.values, d.PatientId.values

    # 5-fold CV, each patient kept in one fold
    rows = []
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
        pw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        for name, model in make_models(pw, CATEGORICAL, NUMERIC).items():
            model.fit(X.iloc[tr], y[tr])
            rows.append(dict(model=name, fold=fold, **scores(y[te], model.predict_proba(X.iloc[te])[:, 1])))
    cv_df = pd.DataFrame(rows)
    cv_df.round(10).to_csv(os.path.join(OUT, 'cv_folds.csv'), index=False)
    cv_sum = cv_df.groupby('model')[['roc_auc', 'pr_auc', 'brier']].agg(['mean', 'std']).round(4)
    cv_sum.to_csv(os.path.join(OUT, 'cv_summary.csv'))
    print('\nPatient-grouped 5-fold CV (mean, std)\n', cv_sum)

    # train before 20 May 2016, test after
    cut = pd.Timestamp('2016-05-20')
    tr, te = d[d.appt < cut], d[d.appt >= cut]
    pw = (tr.y == 0).sum() / max(1, tr.y.sum())
    hold, probs = [], {}
    for name, model in make_models(pw, CATEGORICAL, NUMERIC).items():
        model.fit(tr[NUMERIC + CATEGORICAL], tr.y)
        p = model.predict_proba(te[NUMERIC + CATEGORICAL])[:, 1]
        probs[name] = p
        hold.append(dict(setting='temporal', model=name, **scores(te.y.values, p), **recall_at_fpr(te.y.values, p)))

    # same split without the history features
    no_hist = [c for c in NUMERIC if not c.startswith('prior_')]
    for name, model in make_models(pw, CATEGORICAL, no_hist).items():
        model.fit(tr[no_hist + CATEGORICAL], tr.y)
        p = model.predict_proba(te[no_hist + CATEGORICAL])[:, 1]
        hold.append(dict(setting='temporal, no history', model=name, **scores(te.y.values, p)))

    hold_df = pd.DataFrame(hold).round(4)
    hold_df.to_csv(os.path.join(OUT, 'temporal_holdout.csv'), index=False)
    np.savez(os.path.join(OUT, 'temporal_holdout_probs.npz'), y=te.y.values, **probs)
    print('\nTemporal hold-out\n', hold_df[['setting', 'model', 'roc_auc', 'pr_auc', 'prevalence', 'n',
                                           'recall@fpr5', 'recall@fpr10', 'recall@fpr20']].to_string(index=False))

    summary = dict(n_rows=len(d), n_noshow=int(d.y.sum()), noshow_rate=float(d.y.mean()),
                   n_patients=int(d.PatientId.nunique()), features_numeric=NUMERIC,
                   features_categorical=CATEGORICAL, excluded=EXCLUDED,
                   temporal_split=dict(cut=str(cut.date()), train_n=len(tr), test_n=len(te),
                                       train_rate=float(tr.y.mean()), test_rate=float(te.y.mean())),
                   runtime_sec=round(time.time() - t0, 1))
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']}s. Results in {OUT}")


if __name__ == '__main__':
    main()
