"""
Hotel no-show model (Tables 5 and 6).
Target: No-Show = 1, Check-Out = 0. Cancellations are dropped, they free the room early.
Run: python3 01_noshow_model.py  -> ../results/noshow/
"""
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from common import RANDOM_STATE, make_models, recall_at_fpr, scores
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data', 'hotel_bookings.csv')
OUT = os.path.join(HERE, '..', 'results', 'noshow')
os.makedirs(OUT, exist_ok=True)

# only features known when the booking is made
NUMERIC = ['lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month', 'arrival_month',
           'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
           'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'adr']
CATEGORICAL = ['hotel', 'meal', 'market_segment', 'distribution_channel',
               'reserved_room_type', 'deposit_type', 'customer_type']

# left out, and why (Table 3)
EXCLUDED = {
    'is_canceled': 'target-derived',
    'reservation_status': 'outcome (encodes the label)',
    'reservation_status_date': 'outcome date (set when the booking is resolved)',
    'assigned_room_type': 'assigned at check-in, after the prediction point',
    'booking_changes': 'accumulates between creation and arrival, not known at creation',
    'days_in_waiting_list': 'accumulates after the booking is submitted',
    'required_car_parking_spaces': 'typically recorded at arrival; non-zero almost only for arrivals',
    'total_of_special_requests': 'can be added or changed after creation',
    'country': 'often completed at check-in in the source data (Antonio et al., 2019)',
    'agent': 'identifier (334 levels), not a behavioural feature',
    'company': 'identifier, 94% missing',
    'arrival_date_year': 'calendar year; would not transfer to a new deployment period',
}

MONTHS = {m: i + 1 for i, m in enumerate(['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                          'August', 'September', 'October', 'November', 'December'])}


def load(target='noshow'):
    df = pd.read_csv(DATA)
    df['children'] = df['children'].fillna(0)
    df['arrival_month'] = df['arrival_date_month'].map(MONTHS)
    df['arrival_date'] = pd.to_datetime(dict(year=df.arrival_date_year, month=df.arrival_month,
                                             day=df.arrival_date_day_of_month))
    if target == 'noshow':
        df = df[df.reservation_status.isin(['No-Show', 'Check-Out'])].copy()
        df['y'] = (df.reservation_status == 'No-Show').astype(int)
    else:  # old cancellation target, for comparison only
        df['y'] = df['is_canceled'].astype(int)
    return df.reset_index(drop=True)


def run_cv(df, cat_cols, num_cols, label):
    X, y = df[num_cols + cat_cols], df['y'].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        pw = (y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())
        for name, model in make_models(pw, cat_cols, num_cols).items():
            model.fit(X.iloc[tr], y[tr])
            p = model.predict_proba(X.iloc[te])[:, 1]
            rows.append(dict(setting=label, model=name, fold=fold, **scores(y[te], p)))
    return rows


def run_split(df_tr, df_te, cat_cols, num_cols, label, keep_probs=False):
    y_tr, y_te = df_tr['y'].values, df_te['y'].values
    pw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
    rows, probs = [], {}
    for name, model in make_models(pw, cat_cols, num_cols).items():
        model.fit(df_tr[num_cols + cat_cols], y_tr)
        p = model.predict_proba(df_te[num_cols + cat_cols])[:, 1]
        rows.append(dict(setting=label, model=name, **scores(y_te, p), **recall_at_fpr(y_te, p)))
        if keep_probs:
            probs[name] = p
    return rows, probs


def main():
    t0 = time.time()
    df = load('noshow')
    cat_no_dep = [c for c in CATEGORICAL if c != 'deposit_type']
    feature_sets = {'booking-time': CATEGORICAL, 'booking-time minus deposit_type': cat_no_dep}

    summary = dict(n_rows=len(df), n_noshow=int(df.y.sum()), noshow_rate=float(df.y.mean()),
                   features_numeric=NUMERIC, features_categorical=CATEGORICAL, excluded=EXCLUDED)
    print(f"No-show task: {len(df):,} bookings, {df.y.sum():,} no-shows ({df.y.mean()*100:.2f}%)")

    # 5-fold CV, with and without deposit_type
    cv_rows = []
    for fs_name, cats in feature_sets.items():
        cv_rows += run_cv(df, cats, NUMERIC, fs_name)
    cv = pd.DataFrame(cv_rows)
    cv.round(10).to_csv(os.path.join(OUT, 'cv_folds.csv'), index=False)
    cv_sum = cv.groupby(['setting', 'model'])[['roc_auc', 'pr_auc', 'brier']].agg(['mean', 'std']).round(4)
    cv_sum.to_csv(os.path.join(OUT, 'cv_summary.csv'))
    print('\n5-fold CV (mean, std)\n', cv_sum)

    # train on 2015-16, test on 2017
    cut = pd.Timestamp('2017-01-01')
    tr, te = df[df.arrival_date < cut], df[df.arrival_date >= cut]
    hold_rows, probs = [], {}
    for fs_name, cats in feature_sets.items():
        r, p = run_split(tr, te, cats, NUMERIC, f'temporal | {fs_name}', keep_probs=(fs_name == 'booking-time'))
        hold_rows += r
        probs.update(p)
    np.savez(os.path.join(OUT, 'temporal_holdout_probs.npz'), y=te['y'].values, **probs)

    # train on one hotel, test on the other
    city, resort = df[df.hotel == 'City Hotel'], df[df.hotel == 'Resort Hotel']
    cats_nohotel = [c for c in CATEGORICAL if c != 'hotel']
    hold_rows += run_split(city, resort, cats_nohotel, NUMERIC, 'transfer City->Resort')[0]
    hold_rows += run_split(resort, city, cats_nohotel, NUMERIC, 'transfer Resort->City')[0]

    hold = pd.DataFrame(hold_rows).round(4)
    hold.to_csv(os.path.join(OUT, 'holdout_and_transfer.csv'), index=False)
    print('\nTemporal hold-out and cross-hotel transfer\n', hold.to_string(index=False))

    # cancellation target, comparison only
    canc = run_cv(load('cancel'), CATEGORICAL, NUMERIC, 'cancellation target (comparison)')
    canc_sum = pd.DataFrame(canc).groupby('model')[['roc_auc', 'pr_auc']].agg(['mean', 'std']).round(4)
    canc_sum.to_csv(os.path.join(OUT, 'cancellation_comparison.csv'))
    print('\nCancellation target, booking-time features (comparison only)\n', canc_sum)

    summary['temporal_split'] = dict(train_n=len(tr), test_n=len(te), train_rate=float(tr.y.mean()),
                                     test_rate=float(te.y.mean()))
    summary['runtime_sec'] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone in {summary['runtime_sec']}s. Results in {OUT}")


if __name__ == '__main__':
    main()
