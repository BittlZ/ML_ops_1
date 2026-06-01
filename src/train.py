from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config
from .preprocess import FeaturePipeline


def _holdout_split(groups: np.ndarray, y: np.ndarray):
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    tr_idx, va_idx = next(iter(gkf.split(np.arange(len(y)), y, groups)))
    return tr_idx, va_idx


def main() -> int:
    import catboost
    import lightgbm as lgb
    import xgboost as xgb

    t0 = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[обучение] Загрузка обучающих данных: {config.TRAIN_CSV}")
    train = pd.read_csv(config.TRAIN_CSV)
    print(f"[обучение] Загружено {len(train)} строк.")

    print("[обучение] Обучение пайплайна признаков...")
    pipe = FeaturePipeline().fit(train)
    data = pipe.build_training_matrix(train)
    X_cat, X_le, X_rf = data["cat"], data["le"], data["rf"]
    y, groups = data["y"], data["groups"]
    print(f"[обучение] Матрица признаков: {X_cat.shape[1]} признаков.")

    tr, va = _holdout_split(groups, y)
    seed = config.SEED

    print("[обучение] Обучение CatBoost (CB_deep)...")
    cb = catboost.CatBoostRegressor(
        cat_features=config.CAT_FEATURES, random_seed=seed, verbose=200,
        iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
        early_stopping_rounds=100, eval_metric="RMSE", loss_function="RMSE")
    cb.fit(X_cat.iloc[tr], y[tr], eval_set=(X_cat.iloc[va], y[va]), use_best_model=True)
    cb.save_model(str(config.MODELS_DIR / config.CB_FILE))

    print("[обучение] Обучение XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.03, max_depth=8,
        reg_lambda=3, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, verbosity=0, tree_method="hist",
        early_stopping_rounds=100, n_jobs=-1)
    xgb_model.fit(X_le.iloc[tr], y[tr],
                  eval_set=[(X_le.iloc[va], y[va])], verbose=False)
    xgb_model.save_model(str(config.MODELS_DIR / config.XGB_FILE))

    def train_lgb_quantile(alpha, fname):
        print(f"[обучение] Обучение квантильной модели LightGBM, alpha={alpha}...")
        m = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.03, max_depth=7,
            num_leaves=63, min_child_samples=20, reg_lambda=3,
            objective="quantile", alpha=alpha,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=seed, verbose=-1, n_jobs=-1)
        m.fit(X_le.iloc[tr], y[tr], eval_set=[(X_le.iloc[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
        m.booster_.save_model(str(config.MODELS_DIR / fname))

    train_lgb_quantile(0.5, config.LGB_Q50_FILE)
    train_lgb_quantile(0.7, config.LGB_Q70_FILE)

    print("[обучение] Обучение Hurdle: классификатор + регрессор...")
    y_zero = (y == 0).astype(int)
    clf = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=7,
        num_leaves=63, min_child_samples=20, reg_lambda=3,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1, n_jobs=-1)
    clf.fit(X_le.iloc[tr], y_zero[tr], eval_set=[(X_le.iloc[va], y_zero[va])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    clf.booster_.save_model(str(config.MODELS_DIR / config.HURDLE_CLF_FILE))

    nz = tr[y[tr] > 0]
    reg = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=7,
        num_leaves=63, min_child_samples=20, reg_lambda=3,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, verbose=-1, n_jobs=-1)
    reg.fit(X_le.iloc[nz], y[nz], eval_set=[(X_le.iloc[va], y[va])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    reg.booster_.save_model(str(config.MODELS_DIR / config.HURDLE_REG_FILE))

    print("[обучение] Обучение ExtraTrees...")
    from sklearn.ensemble import ExtraTreesRegressor
    # Fewer trees + min_samples_leaf keep the serialised model well under
    # GitHub's 100 MB file limit; compress=3 shrinks it further. ET only carries
    # ~9% ensemble weight, so this has negligible impact on the blend.
    et = ExtraTreesRegressor(n_estimators=300, max_depth=18, min_samples_leaf=10,
                             max_features=0.7, n_jobs=-1, random_state=seed)
    et.fit(X_rf, y)
    joblib.dump(et, config.MODELS_DIR / config.ET_FILE, compress=3)

    print("[обучение] Сохранение пайплайна признаков и метаданных...")
    joblib.dump(pipe, config.MODELS_DIR / config.PIPELINE_FILE)
    meta = {
        "ensemble_weights": config.ENSEMBLE_WEIGHTS,
        "n_features": int(X_cat.shape[1]),
        "feature_columns": pipe.feature_columns,
        "cat_features": config.CAT_FEATURES,
        "trained_on_rows": int(len(train)),
    }
    with open(config.MODELS_DIR / config.META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[обучение] Готово за {time.time() - t0:.0f}с. Артефакты в {config.MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
