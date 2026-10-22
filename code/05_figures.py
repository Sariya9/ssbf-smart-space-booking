"""
Draws the paper's figures from the saved results (run after 01-04 and 08).
Run: python3 05_figures.py  -> ../figures/
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler

import importlib.util
from common import RANDOM_STATE, make_models

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, GREY, INK = '#2a78d6', '#eb6834', '#1baf7a', '#8a8a85', '#2b2b29'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': GREY,
                     'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e5e0', 'grid.linewidth': 0.6, 'lines.linewidth': 2,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.axisbelow': True})


def load_module(fname):
    spec = importlib.util.spec_from_file_location(fname[:-3].replace('0', 'm', 1), os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def roc_figure():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (name, path) in zip(axes, [('Hotel (true no-shows), 2017 hold-out', 'noshow'),
                                        ('Medical appointments, late-May hold-out', 'medical')]):
        d = np.load(os.path.join(RES, path, 'temporal_holdout_probs.npz'))
        for model, col in [('RF', BLUE), ('XGB', ORANGE), ('LR', AQUA)]:
            fpr, tpr, _ = roc_curve(d['y'], d[model])
            from sklearn.metrics import roc_auc_score
            ax.plot(fpr, tpr, color=col, label=f'{model} (AUC {roc_auc_score(d["y"], d[model]):.3f})')
        ax.plot([0, 1], [0, 1], '--', color=GREY, linewidth=1, label='Chance')
        ax.set(title=name, xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1))
        ax.title.set_fontsize(9)
        ax.legend(frameon=False, loc='lower right', fontsize=8)
    fig.savefig(os.path.join(FIG, 'r1_noshow_roc.png'))
    plt.close(fig)


def shap_for(df_tr, df_te, cats, nums, label, n=1000):
    pw = (df_tr.y == 0).sum() / max(1, df_tr.y.sum())
    pipe = make_models(pw, cats, nums)['RF']
    pipe.fit(df_tr[nums + cats], df_tr.y)
    prep, clf = pipe.named_steps['prep'], pipe.named_steps['clf']
    names = nums + list(prep.named_transformers_['cat'].get_feature_names_out(cats))
    sample = df_te.sample(min(n, len(df_te)), random_state=RANDOM_STATE)
    Xs = prep.transform(sample[nums + cats])
    Xs = Xs.toarray() if hasattr(Xs, 'toarray') else Xs
    sv = shap.TreeExplainer(clf).shap_values(Xs)
    sv = sv[:, :, 1] if np.ndim(sv) == 3 else sv[1]
    ev = shap.TreeExplainer(clf).expected_value
    ev = ev[1] if np.ndim(ev) else ev
    return dict(label=label, sv=sv, X=Xs, names=names, base=ev, probs=pipe.predict_proba(sample[nums + cats])[:, 1],
                y=sample.y.values)


def shap_figures():
    m1 = load_module('01_noshow_model.py')
    m2 = load_module('02_medical_noshow_model.py')
    h = m1.load('noshow')
    cut = pd.Timestamp('2017-01-01')
    hotel = shap_for(h[h.arrival_date < cut], h[h.arrival_date >= cut], m1.CATEGORICAL, m1.NUMERIC, 'Hotel')
    d = m2.load()
    cut = pd.Timestamp('2016-05-20')
    med = shap_for(d[d.appt < cut], d[d.appt >= cut], m2.CATEGORICAL, m2.NUMERIC, 'Medical')

    # top features, hotel and medical
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    rows = []
    for ax, s in zip(axes, [hotel, med]):
        imp = pd.Series(np.abs(s['sv']).mean(0), index=s['names']).sort_values().tail(10)
        ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
        ax.set(title=f"{s['label']} RF: top 10 features", xlabel='Mean |SHAP| (no-show probability)')
        ax.title.set_fontsize(9)
        ax.grid(axis='y', visible=False)
        for k, v in imp.sort_values(ascending=False).items():
            rows.append(dict(domain=s['label'], feature=k, mean_abs_shap=round(float(v), 5)))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'r1_shap_bar.png'))
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'shap_top_features.csv'), index=False)

    # SHAP summary for the medical model
    plt.figure()
    np.random.seed(RANDOM_STATE)        # fixes the dot jitter
    shap.summary_plot(med['sv'], med['X'], feature_names=med['names'], max_display=12, show=False,
                      plot_size=(7, 4.6))
    plt.savefig(os.path.join(FIG, 'r1_shap_summary_medical.png'))
    plt.close('all')

    # one example: the highest-risk real no-show
    i = int(np.argmax(np.where(med['y'] == 1, med['probs'], -1)))
    expl = shap.Explanation(values=med['sv'][i], base_values=med['base'], data=med['X'][i],
                            feature_names=med['names'])
    plt.figure()
    shap.plots.waterfall(expl, max_display=10, show=False)
    plt.gcf().set_size_inches(7, 4.2)
    plt.savefig(os.path.join(FIG, 'r1_shap_waterfall_medical.png'))
    plt.close('all')
    with open(os.path.join(RES, 'shap_waterfall_case.txt'), 'w') as f:
        f.write(f"predicted={med['probs'][i]:.3f} base={med['base']:.3f}\n")
        for k, v, x in sorted(zip(med['names'], med['sv'][i], med['X'][i]), key=lambda t: -abs(t[1]))[:10]:
            f.write(f'{k}\tvalue={x:.3f}\tshap={v:+.4f}\n')


def occupancy_figures():
    path = os.path.join(HERE, '..', 'data', 'occupancy+detection')
    train = pd.read_csv(os.path.join(path, 'datatraining.txt'))
    test2 = pd.read_csv(os.path.join(path, 'datatest2.txt'))
    sensors = ['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']
    sc = StandardScaler().fit(train[sensors])
    lr = LogisticRegression(max_iter=1000, class_weight='balanced').fit(sc.transform(train[sensors]), train.Occupancy)
    p = lr.predict_proba(sc.transform(test2[sensors]))[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    cm = confusion_matrix(test2.Occupancy, (p >= 0.5).astype(int))
    ConfusionMatrixDisplay(cm, display_labels=['Vacant', 'Occupied']).plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].grid(False)
    axes[0].set_title('LR confusion matrix, Test Set 2 (n = 9,752)', fontsize=9)
    fpr, tpr, _ = roc_curve(test2.Occupancy, p)
    from sklearn.metrics import roc_auc_score
    axes[1].plot(fpr, tpr, color=BLUE, label=f'LR (AUC {roc_auc_score(test2.Occupancy, p):.4f})')
    axes[1].plot([0, 1], [0, 1], '--', color=GREY, linewidth=1, label='Chance')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC, Test Set 2')
    axes[1].title.set_fontsize(9)
    axes[1].legend(frameon=False, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'r1_uci_lr.png'))
    plt.close(fig)

    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, class_weight='balanced',
                                random_state=RANDOM_STATE, n_jobs=-1).fit(train[sensors], train.Occupancy)
    imp = pd.Series(rf.feature_importances_, index=sensors).sort_values()
    coef = pd.Series(lr.coef_[0], index=sensors).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    axes[0].barh(imp.index, imp.values, color=BLUE, height=0.6)
    axes[0].set(title='RF impurity importance', xlabel='Share of importance')
    axes[1].barh(coef.index, coef.values, color=[ORANGE if v < 0 else BLUE for v in coef.values], height=0.6)
    axes[1].axvline(0, color=GREY, linewidth=1)
    axes[1].set(title='LR standardised coefficients', xlabel='Change in log-odds per 1 SD')
    for ax in axes:
        ax.title.set_fontsize(9)
        ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'r1_uci_sensors.png'))
    plt.close(fig)
    pd.DataFrame({'rf_importance': imp, 'lr_coef': coef, 'odds_ratio': np.exp(coef)}).round(4).to_csv(
        os.path.join(RES, 'occupancy', 'sensor_importance.csv'))


def sensor_timeseries_figure():
    # Fig. 2: two days of room A, occupancy against CO2 and light
    path = os.path.join(HERE, '..', 'data', 'occupancy+detection')
    d = pd.read_csv(os.path.join(path, 'datatraining.txt'), parse_dates=['date'])
    d = d[(d.date >= '2015-02-05') & (d.date < '2015-02-07')]   # Thu and Fri
    fig, axes = plt.subplots(3, 1, figsize=(8, 5.2), sharex=True)
    axes[0].plot(d.date, d.Occupancy, color=INK, linewidth=1.2, drawstyle='steps-post')
    axes[0].set(ylabel='Occupied', yticks=[0, 1], yticklabels=['No', 'Yes'])
    axes[1].plot(d.date, d.CO2, color=ORANGE, linewidth=1.2)
    axes[1].set(ylabel='CO$_2$ (ppm)')
    axes[2].plot(d.date, d.Light, color=BLUE, linewidth=1.2)
    axes[2].set(ylabel='Light (lux)')
    axes[2].xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%a %H:%M'))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'fig12_sensor_timeseries.png'))
    plt.close(fig)


def simulation_figures():
    g = pd.read_csv(os.path.join(RES, 'simulation', 'sensitivity_gains.csv'))
    panels = [('rho', 'Demand intensity (requests per room-slot)'), ('accept_prob', 'Offer acceptance probability'),
              ('noshow_mean', 'Mean no-show rate'), ('walkin_rate', 'Walk-in claimants per released room')]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharey=False)
    for ax, (param, xlabel) in zip(axes.flat, panels):
        s = g[g.param == param].sort_values('value')
        ax.errorbar(s.value, s.gain_vs_none, yerr=[s.gain_vs_none - s.ci_low, s.ci_high - s.gain_vs_none],
                    color=BLUE, marker='o', markersize=5, capsize=2, label='SSBF')
        ax.errorbar(s.value, s.ext_gain, yerr=[s.ext_gain - s.ext_ci_low, s.ext_ci_high - s.ext_gain],
                    color=ORANGE, marker='s', markersize=5, capsize=2, label='SSBF + extended redistribution')
        ax.set(xlabel=xlabel, ylabel='Utilisation gain over no release (pp)')
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'r1_sim_sensitivity.png'))
    plt.close(fig)

    runs = pd.read_csv(os.path.join(RES, 'simulation', 'main_runs.csv'))
    order = ['ML only + DRF', 'Check-in release + FCFS', 'Sensing + FCFS', 'Sensing + WFQ', 'Sensing + DRF',
             'ML + Sensing + DRF (SSBF)', 'Oracle ML + Sensing + DRF (upper bound)', 'SSBF + extended redistribution']
    agg = runs.groupby('condition').recovered_pct.agg(['mean', 'std', 'count']).loc[order]
    half = 2.045 * agg['std'] / np.sqrt(agg['count'])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colours = [GREY] * 5 + [BLUE, GREY, ORANGE]
    ax.barh(range(len(order)), agg['mean'], xerr=half, color=colours, height=0.6, capsize=2,
            error_kw=dict(ecolor=INK, elinewidth=1))
    ax.set_yticks(range(len(order)), [o.replace(' (upper bound)', '\n(upper bound)') for o in order], fontsize=8)
    for k, v in enumerate(agg['mean']):
        ax.text(v + half.iloc[k] + 0.4, k, f'{v:.1f}%', va='center', fontsize=8, color=INK)
    ax.set(xlabel='Share of room time lost to ghost bookings and late arrivals that is recovered (%)')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, agg['mean'].max() * 1.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'r1_sim_recovery.png'))
    plt.close(fig)


if __name__ == '__main__':
    roc_figure()
    occupancy_figures()
    sensor_timeseries_figure()
    simulation_figures()
    shap_figures()
    print('Figures written to', os.path.abspath(FIG))
