"""Shared models and metrics for the prediction scripts."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42   # fixed seed


def make_models(pos_weight, cat_cols, num_cols):
    # class weights instead of SMOTE; pos_weight = negatives/positives (XGB)
    def prep(scale):
        # only LR needs scaling
        num = StandardScaler() if scale else 'passthrough'
        return ColumnTransformer([('num', num, num_cols),
                                  ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])
    return {
        'LR': Pipeline([('prep', prep(True)),
                        ('clf', LogisticRegression(C=1.0, solver='liblinear', max_iter=1000,
                                                   class_weight='balanced'))]),
        'RF': Pipeline([('prep', prep(False)),
                        ('clf', RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                                       class_weight='balanced_subsample',
                                                       n_jobs=-1, random_state=RANDOM_STATE))]),
        'XGB': Pipeline([('prep', prep(False)),
                         ('clf', XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                                               scale_pos_weight=pos_weight, eval_metric='logloss',
                                               n_jobs=-1, random_state=RANDOM_STATE))]),
    }


def scores(y, p):
    return dict(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), prevalence=float(np.mean(y)), n=int(len(y)))


def recall_at_fpr(y, p, fprs=(0.01, 0.05, 0.10, 0.20)):
    # recall at a capped false-alarm rate; feeds the simulation
    fpr, tpr, thr = roc_curve(y, p)
    out = {}
    for f in fprs:
        i = np.searchsorted(fpr, f, side='right') - 1
        out[f'recall@fpr{int(f*100)}'] = float(tpr[i])
        out[f'threshold@fpr{int(f*100)}'] = float(thr[i])
    return out
