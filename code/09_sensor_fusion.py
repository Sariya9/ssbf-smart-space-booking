"""
Does adding more sensor types help? (Table 12, part of Table 11)
ROBOD: light, CO2, temperature, then humidity and Wi-Fi, then sound, VOC and PM2.5, then the
lighting and plug-load meters, plus one set with no light at all.
Room B: light, CO2, temperature, then PIR, sound and CO2 slope (within room only, it has no twin).
Room A has none of the extra sensors, so it is not used here.
Each set: within room, other rooms with no set-up, and after k set-up days (as in 08).
Also runs the simulation with the error rates of three of these sets.
Run: python3 09_sensor_fusion.py  -> ../results/occupancy/, ../results/simulation/
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'occupancy')


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


building = _load('building', '07_building_occupancy.py')
crossroom = _load('crossroom', '06_cross_room_occupancy.py')

TARGET_FV = 0.02
K_DAYS = [0, 1, 2, 3, 5, 7]

ROBOD_SETS = {
    'Light + CO2 + temperature (original)': ['Light', 'CO2', 'Temperature'],
    'Environmental + Wi-Fi': ['Light', 'CO2', 'Temperature', 'Humidity', 'WiFi'],
    'All room sensors': ['Light', 'CO2', 'Temperature', 'Humidity', 'WiFi', 'Sound', 'VOC', 'PM25'],
    'All room sensors + building meters': ['Light', 'CO2', 'Temperature', 'Humidity', 'WiFi', 'Sound', 'VOC', 'PM25',
                                           'LightingEnergy', 'PlugEnergy'],
    'No light: CO2 + Wi-Fi + sound': ['CO2', 'WiFi', 'Sound'],
}
ROOMB_SETS = {
    'Light + CO2 + temperature (original)': ['Light', 'CO2', 'Temperature'],
    'Light + CO2 + temperature + PIR': ['Light', 'CO2', 'Temperature', 'PIR'],
    'All sensors (+ PIR, sound, CO2 slope)': ['Light', 'CO2', 'Temperature', 'PIR', 'Sound', 'CO2_slope'],
    'No light: CO2 + PIR + sound': ['CO2', 'PIR', 'Sound'],
}


def models():
    return {'LR': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced')),
            'RF': RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced',
                                         n_jobs=1, random_state=42)}


def rates(y, p, thr=0.5):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return fn / max(1, fn + tp), fp / max(1, fp + tn)


def robod_rooms():
    rooms = building.load_robod()
    for d in rooms.values():
        d['Sound'] = d['sound_pressure_level [dba]']
        d['VOC'] = d['voc [ppb]']
        d['PM25'] = d['pm2.5 [mu_g/m3]']
    for r, d in rooms.items():
        cols = sorted({c for v in ROBOD_SETS.values() for c in v})
        rooms[r] = d.dropna(subset=cols).reset_index(drop=True)
    return rooms


def split_days(d):
    days = np.sort(d.ts.dt.normalize().unique())
    cut = days[int(len(days) * 0.7)]
    return d[d.ts < cut], d[d.ts >= cut], [x for x in days if x < cut]


def main():
    rows = []
    rooms = robod_rooms()
    for fs, cols in ROBOD_SETS.items():
        for r, d in rooms.items():
            train, test, pre_days = split_days(d)
            others = pd.concat([rooms[o] for o in rooms if o != r])
            for mname, make in [('LR', lambda: models()['LR']), ('RF', lambda: models()['RF'])]:
                tag = dict(dataset='ROBOD', features=fs, room=building.ROOMS[r], model=mname)
                m = make().fit(train[cols], train.occ)
                p = m.predict_proba(test[cols])[:, 1]
                fv, mv = rates(test.occ.values, p)
                rows.append(dict(setting='within room', k_days=np.nan, auc=roc_auc_score(test.occ, p),
                                 false_vacancy=fv, missed_vacancy=mv, **tag))
                src = make().fit(others[cols], others.occ)
                p = src.predict_proba(test[cols])[:, 1]
                fv, mv = rates(test.occ.values, p)
                rows.append(dict(setting='other rooms, no set-up', k_days=0, auc=roc_auc_score(test.occ, p),
                                 false_vacancy=fv, missed_vacancy=mv, **tag))
                for k in K_DAYS[1:]:
                    adapt = d[d.ts.dt.normalize().isin(pre_days[:k])]
                    thr = 0.5
                    if (adapt.occ == 1).any():
                        pa = src.predict_proba(adapt[cols])[:, 1]
                        thr = float(np.quantile(pa[adapt.occ.values == 1], TARGET_FV))
                    fv, mv = rates(test.occ.values, p, thr)
                    rows.append(dict(setting='other rooms, re-threshold', k_days=k, auc=roc_auc_score(test.occ, p),
                                     false_vacancy=fv, missed_vacancy=mv, **tag))

    # room B: within room only
    _, _, b, b_train, b_test = crossroom.load_rooms()
    b = b.copy()
    for d in (b, b_train, b_test):
        d['Sound'] = d[['S1_Sound', 'S2_Sound', 'S3_Sound', 'S4_Sound']].mean(axis=1)
        d['CO2_slope'] = d['S5_CO2_Slope']
    for fs, cols in ROOMB_SETS.items():
        for mname in ('LR', 'RF'):
            m = models()[mname].fit(b_train[cols], b_train.occ)
            p = m.predict_proba(b_test[cols])[:, 1]
            fv, mv = rates(b_test.occ.values, p)
            rows.append(dict(dataset='Room B', features=fs, room='Room B', model=mname, setting='within room (Dec -> Jan)',
                             k_days=np.nan, auc=roc_auc_score(b_test.occ, p), false_vacancy=fv, missed_vacancy=mv))

    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(OUT, 'fusion.csv'), index=False)
    summ = res.groupby(['dataset', 'features', 'model', 'setting', 'k_days'], sort=False, dropna=False).agg(
        auc_median=('auc', 'median'), fv_median=('false_vacancy', 'median'), fv_max=('false_vacancy', 'max'),
        mv_median=('missed_vacancy', 'median'), rooms=('room', 'count')).round(3).reset_index()
    summ.to_csv(os.path.join(OUT, 'fusion_summary.csv'), index=False)
    pd.set_option('display.width', 220)
    show = summ[(summ.k_days.isna()) | (summ.k_days.isin([0, 3]))]
    print(show.to_string(index=False))
    simulate_scenarios(summ)


def simulate_scenarios(summ, reps=30):
    # simulation with the measured error rates (LR, median over ROBOD rooms)
    sim = _load('sim', '03_queue_simulation.py')
    def get(fs, setting, k=None):
        x = summ[(summ.dataset == 'ROBOD') & (summ.features == fs) & (summ.setting == setting) & (summ.model == 'LR')]
        if k is not None:
            x = x[x.k_days == k]
        x = x.iloc[0]
        return dict(sensor_false_vacant=float(x.fv_median), sensor_outage=float(x.mv_median))
    scenarios = {
        'original sensors, own room': get('Light + CO2 + temperature (original)', 'within room'),
        'all sensors + meters, own room': get('All room sensors + building meters', 'within room'),
        'all room sensors, new room, 3-day re-threshold': get('All room sensors', 'other rooms, re-threshold', 3),
    }
    conds = ['No release', 'Check-in release + FCFS', 'ML + Sensing + DRF (SSBF)', 'SSBF + extended redistribution']
    rows = []
    for name, ov in scenarios.items():
        p = dict(sim.DEFAULTS, **ov)
        for seed in range(1000, 1000 + reps):
            ev = sim.generate(p, seed)
            res = {c: sim.simulate(c, ev, p) for c in conds}
            base, booked = res['No release']['utilisation'], res['No release']['booked_share']
            for c in conds[1:]:
                r = res[c]
                rows.append(dict(scenario=name, condition=c, seed=seed, **ov,
                                 recovered_pct=(r['utilisation'] - base) / (booked - base) * 100,
                                 wrongful_per100=r['wrongful_per100'], displaced_per100=r['displaced_per100'],
                                 futile_per100=r['futile_per100']))
    out = pd.DataFrame(rows).groupby(['scenario', 'condition'], sort=False)[
        ['sensor_false_vacant', 'sensor_outage', 'recovered_pct', 'wrongful_per100', 'displaced_per100',
         'futile_per100']].mean().round(3)
    out.to_csv(os.path.join(HERE, '..', 'results', 'simulation', 'fusion_scenarios.csv'))
    print(out.to_string())


if __name__ == '__main__':
    main()
