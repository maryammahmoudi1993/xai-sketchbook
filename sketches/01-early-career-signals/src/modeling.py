"""Temporal split, logistic baseline, LightGBM main model, bootstrap metrics, calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import FEATURES_ALL, FEATURES_NO_EARLY_CITES

TRAIN_YEARS = (2008, 2009)
TEST_YEARS = (2010, 2011)


def temporal_split(df: pd.DataFrame):
    tr = df[df.cohort_year.between(*TRAIN_YEARS)]
    te = df[df.cohort_year.between(*TEST_YEARS)]
    return tr, te


def make_logistic(features: list[str]):
    cat = ["institution_tier"] if "institution_tier" in features else []
    num = [f for f in features if f not in cat]
    ct = ColumnTransformer([("num", StandardScaler(), num),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(ct, LogisticRegression(max_iter=2000, C=1.0, random_state=config.SEED))


def make_lgbm(pos_rate: float):
    return LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          scale_pos_weight=(1 - pos_rate) / max(pos_rate, 1e-6),
                          random_state=config.SEED, verbose=-1)


def bootstrap_ci(y, p, metric, n_boot: int = 1000, seed: int = config.SEED):
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(metric(y[i], p[i]))
    return float(metric(y, p)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(y, p, n_boot: int = 1000) -> dict:
    auc = bootstrap_ci(y, p, roc_auc_score, n_boot)
    ap = bootstrap_ci(y, p, average_precision_score, n_boot)
    return {"roc_auc": auc, "pr_auc": ap, "base_rate": float(np.mean(y)), "n": int(len(y)),
            "n_pos": int(np.sum(y))}


def platt_on_oof(model, X_tr, y_tr, X_te):
    """Recalibrate class-weighted LightGBM scores: Platt scaling fitted on out-of-fold train scores."""
    skf = StratifiedKFold(3, shuffle=True, random_state=config.SEED)
    oof = cross_val_predict(model, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    cal = LogisticRegression().fit(lg(oof), y_tr)
    model.fit(X_tr, y_tr)
    return cal.predict_proba(lg(model.predict_proba(X_te)[:, 1]))[:, 1], model


def run_all(df: pd.DataFrame, n_boot: int = 1000) -> dict:
    """df has FEATURES_ALL + label. Returns metrics, test predictions and fitted LGBM models."""
    tr, te = temporal_split(df)
    out = {"n_train": len(tr), "n_test": len(te), "preds": {}, "metrics": {}, "models": {}, "calibration": {}}
    for tag, feats in (("with_early_citations", FEATURES_ALL), ("without_early_citations", FEATURES_NO_EARLY_CITES)):
        lr = make_logistic(feats).fit(tr[feats], tr.label)
        p_lr = lr.predict_proba(te[feats])[:, 1]
        gb = make_lgbm(tr.label.mean())
        p_gb, gb = platt_on_oof(gb, tr[feats], tr.label, te[feats])
        for name, p in (("logistic", p_lr), ("lightgbm", p_gb)):
            out["metrics"][(name, tag)] = evaluate(te.label, p, n_boot)
            out["preds"][(name, tag)] = p
            frac, mean_p = calibration_curve(te.label, p, n_bins=8, strategy="quantile")
            out["calibration"][(name, tag)] = (mean_p, frac)
        out["models"][tag] = gb
    out["test"] = te
    out["train"] = tr
    return out
