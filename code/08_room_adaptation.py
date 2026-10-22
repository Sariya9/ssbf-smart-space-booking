"""
How many labelled days does a new room need? (Fig. 6, Table 11)
Each ROBOD room plays the "new room" in turn. Test = its last 30% of days (same as script 07);
set-up = its first k days (k = 0-7). Starting model: the other four rooms, or room A.
Ways to adapt: keep as is, re-set the threshold (2% target), retrain with the new days,
train on the new days only, or a label-free baseline. Also runs the simulation with the error rates found.
Run: python3 08_room_adaptation.py  -> ../results/occupancy/, ../results/simulation/
"""
import importlib.util
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'results', 'occupancy')
SIM_OUT = os.path.join(HERE, '..', 'results', 'simulation')
FIG = os.path.join(HERE, '..', 'manuscript', 'Dataset', 'Figures')
if not os.path.isdir(FIG):   # public package layout
    FIG = os.path.join(HERE, '..', 'figures')
    os.makedirs(FIG, exist_ok=True)
_spec = importlib.util.spec_from_file_location('building', os.path.join(HERE, '07_building_occupancy.py'))
building = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(building)

K_DAYS = [0, 1, 2, 3, 5, 7]
TARGET_FV = 0.02            # aim: at most 2% of occupied readings called empty
FEATURES = {'Light + CO2 + temperature': ['Light', 'CO2', 'Temperature'],
            'Environmental + Wi-Fi': ['Light', 'CO2', 'Temperature', 'Humidity', 'WiFi']}


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced'))


def rates(y, p, thr=0.5):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return fn / max(1, fn + tp), fp / max(1, fp + tn)


def baseline_relative(df, cols, tcol):
    # label-free: distance above the room's quiet level over the last 24 h
    out = df.copy()
    t = pd.DatetimeIndex(pd.to_datetime(df[tcol]))
    for c in cols:
        roll = pd.Series(df[c].values, index=t).rolling('24h', min_periods=12)
        lo, q1, q3 = roll.quantile(0.05), roll.quantile(0.25), roll.quantile(0.75)
        scale = (q3 - q1).where((q3 - q1) > 1e-6, df[c].std())
        out[c] = ((df[c] - lo.values) / scale.values).fillna(0.0)
    return out


def main():
    rooms = building.load_robod()
    uci_tr, uci_te = building.load_uci()
    uci_all = pd.concat([uci_tr, uci_te]).reset_index(drop=True)
    rows = []
    for fs, cols in FEATURES.items():
        sources = {'same building': None}
        if all(c in uci_all for c in cols):
            sources['other building'] = uci_all
        for r, d in rooms.items():
            days = np.sort(d.ts.dt.normalize().unique())
            cut = days[int(len(days) * 0.7)]
            test = d[d.ts >= cut]
            pre_days = [x for x in days if x < cut]
            for src_name, src in sources.items():
                src_df = pd.concat([rooms[o] for o in rooms if o != r]) if src is None else src
                src_model = lr().fit(src_df[cols], src_df.occ)
                p_src = src_model.predict_proba(test[cols])[:, 1]
                # label-free version, same transform in both rooms
                if src is None:
                    src_rel = pd.concat([baseline_relative(rooms[o], cols, 'ts') for o in rooms if o != r])
                else:
                    src_rel = baseline_relative(src.sort_values('date').reset_index(drop=True), cols, 'date')
                rel_model = lr().fit(src_rel[cols], src_rel.occ)
                tgt_rel = baseline_relative(d, cols, 'ts')
                p_rel = rel_model.predict_proba(tgt_rel.loc[test.index, cols])[:, 1]
                for k in K_DAYS:
                    adapt = d[d.ts.dt.normalize().isin(pre_days[:k])]
                    has_both = k > 0 and adapt.occ.nunique() == 2
                    tag = dict(features=fs, source=src_name, room=building.ROOMS[r], k_days=k,
                               adapt_occupied=int(adapt.occ.sum()) if k else 0)

                    def add(method, p, thr=0.5):
                        fv, mv = rates(test.occ.values, p, thr)
                        rows.append(dict(method=method, auc=roc_auc_score(test.occ, p), false_vacancy=fv,
                                         missed_vacancy=mv, threshold=thr, **tag))

                    if k == 0:
                        add('source only', p_src)
                        add('baseline-relative (label-free)', p_rel)
                        continue
                    # re-set the threshold on the set-up days
                    if (adapt.occ == 1).any():
                        p_ad = src_model.predict_proba(adapt[cols])[:, 1]
                        thr = float(np.quantile(p_ad[adapt.occ.values == 1], TARGET_FV))
                        add('re-threshold', p_src, thr)
                    else:
                        add('re-threshold', p_src)
                    # retrain with the set-up days, weighted to count as much as the source
                    pool = pd.concat([src_df[cols + ['occ']], adapt[cols + ['occ']]])
                    w = np.r_[np.ones(len(src_df)), np.full(len(adapt), len(src_df) / max(1, len(adapt)))]
                    m = lr()
                    m.fit(pool[cols], pool.occ, logisticregression__sample_weight=w)
                    add('pooled retrain', m.predict_proba(test[cols])[:, 1])
                    if has_both and src_name == 'same building':
                        loc = lr().fit(adapt[cols], adapt.occ)
                        add('local only', loc.predict_proba(test[cols])[:, 1])

    res = pd.DataFrame(rows).round(4)
    res.to_csv(os.path.join(OUT, 'adaptation.csv'), index=False)
    summ = res.groupby(['features', 'source', 'method', 'k_days'], sort=False).agg(
        auc_median=('auc', 'median'), fv_median=('false_vacancy', 'median'), fv_max=('false_vacancy', 'max'),
        mv_median=('missed_vacancy', 'median'), rooms=('room', 'count')).round(3).reset_index()
    summ.to_csv(os.path.join(OUT, 'adaptation_summary.csv'), index=False)
    pd.set_option('display.width', 200)
    print(summ.to_string(index=False))
    simulate_scenarios(summ)
    plot(summ)


def simulate_scenarios(summ, reps=30):
    # run the simulation with the error rates we measured (per reading -> per booking)
    spec = importlib.util.spec_from_file_location('sim', os.path.join(HERE, '03_queue_simulation.py'))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    s = summ[summ.features == 'Light + CO2 + temperature']
    def get(src, method, k):
        x = s[(s.source == src) & (s.method == method) & (s.k_days == k)].iloc[0]
        return dict(sensor_false_vacant=float(x.fv_median), sensor_outage=float(x.mv_median))
    scenarios = {'calibrated in its own room (default)': dict(sensor_false_vacant=sim.DEFAULTS['sensor_false_vacant'],
                                                                  sensor_outage=sim.DEFAULTS['sensor_outage']),
                 'new room, no commissioning': get('same building', 'source only', 0),
                 'new room, 3-day re-threshold': get('same building', 're-threshold', 3)}
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
    df = pd.DataFrame(rows)
    out = df.groupby(['scenario', 'condition'], sort=False)[
        ['sensor_false_vacant', 'sensor_outage', 'recovered_pct', 'wrongful_per100', 'displaced_per100',
         'futile_per100']].mean().round(3)
    out.to_csv(os.path.join(SIM_OUT, 'adaptation_scenarios.csv'))
    print(out.to_string())


def plot(summ):
    blue, orange, aqua, grey, ink = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a85', '#2b2b29'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e5e0',
                         'axes.axisbelow': True, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    s = summ[summ.features == 'Light + CO2 + temperature']
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.6), sharex=True)
    lines = [('re-threshold', blue, 'o'), ('pooled retrain', orange, 's'), ('local only', aqua, '^')]
    for col, (src, title) in enumerate([('same building', 'Model from other rooms, same building'),
                                        ('other building', 'Model from another building')]):
        for row, (metric, ylabel) in enumerate([('fv_median', 'Median false vacancy (%)'),
                                                ('mv_median', 'Median missed vacancy (%)')]):
            ax = axes[row, col]
            for method, c, mk in lines:
                x = s[(s.source == src) & (s.method == method)].sort_values('k_days')
                if len(x):
                    ax.plot(x.k_days, x[metric] * 100, color=c, marker=mk, markersize=5, linewidth=2, label=method)
            k0 = s[(s.source == src) & (s.k_days == 0)].set_index('method')[metric] * 100
            ax.scatter([0], [k0['source only']], color=ink, marker='x', s=40, zorder=3, label='no adaptation')
            ax.scatter([0], [k0['baseline-relative (label-free)']], color=grey, marker='D', s=30, zorder=3,
                       label='label-free baseline-relative')
            if metric == 'fv_median':
                ax.axhline(2.0, color=grey, linestyle='--', linewidth=1)
                ax.set_yscale('symlog', linthresh=5)
                ax.set_ylim(0, 110)
                ax.set_yticks([0, 2, 5, 10, 20, 50, 100], ['0', '2', '5', '10', '20', '50', '100'])
                ax.set_title(title, fontsize=9)
            else:
                ax.set_ylim(0, 100)
                ax.set_xlabel('Labelled commissioning days from the new room')
            ax.set_xticks([0, 1, 2, 3, 5, 7])
            if col == 0:
                ax.set_ylabel(ylabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, 'r1_adaptation.png'))
    plt.close(fig)

if __name__ == '__main__':
    main()
