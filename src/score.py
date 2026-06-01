from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config


class ScoringEngine:
    def __init__(self) -> None:
        self.pipeline = None
        self.models = {}

    def load(self) -> "ScoringEngine":
        import catboost
        import lightgbm as lgb
        import xgboost as xgb

        md = config.MODELS_DIR
        print(f"[скоринг] Загрузка артефактов из {md}")

        self.pipeline = joblib.load(md / config.PIPELINE_FILE)

        cb = catboost.CatBoostRegressor()
        cb.load_model(str(md / config.CB_FILE))
        self.models["CB_deep"] = cb

        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(str(md / config.XGB_FILE))
        self.models["XGB"] = xgb_model

        self.models["LGB_q50"] = lgb.Booster(model_file=str(md / config.LGB_Q50_FILE))
        self.models["LGB_q70"] = lgb.Booster(model_file=str(md / config.LGB_Q70_FILE))
        self.models["Hurdle_clf"] = lgb.Booster(model_file=str(md / config.HURDLE_CLF_FILE))
        self.models["Hurdle_reg"] = lgb.Booster(model_file=str(md / config.HURDLE_REG_FILE))
        self.models["ET"] = joblib.load(md / config.ET_FILE)

        print("[скоринг] Все модели успешно загружены.")
        return self

    def _clip(self, arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, config.TARGET_MIN, config.TARGET_MAX)

    def predict(self, views: dict) -> np.ndarray:
        X_cat, X_le, X_rf = views["cat"], views["le"], views["rf"]

        preds = {}
        preds["CB_deep"] = self._clip(self.models["CB_deep"].predict(X_cat))
        preds["XGB"] = self._clip(self.models["XGB"].predict(X_le))
        preds["LGB_q50"] = self._clip(self.models["LGB_q50"].predict(X_le))
        preds["LGB_q70"] = self._clip(self.models["LGB_q70"].predict(X_le))

        p_zero = self.models["Hurdle_clf"].predict(X_le)
        nonzero = self.models["Hurdle_reg"].predict(X_le)
        preds["Hurdle"] = self._clip((1.0 - p_zero) * nonzero)

        preds["ET"] = self._clip(self.models["ET"].predict(X_rf))

        blend = np.zeros(len(X_cat), dtype=float)
        total_w = sum(config.ENSEMBLE_WEIGHTS.values())
        for name, w in config.ENSEMBLE_WEIGHTS.items():
            blend += (w / total_w) * preds[name]
        final = self._clip(blend)

        print("[скоринг] Средние предсказания по моделям: " +
              ", ".join(f"{n}={preds[n].mean():.1f}" for n in preds))
        print(f"[скоринг] Ансамбль: среднее={final.mean():.1f}, "
              f"ст. отклонение={final.std():.1f}")
        return final

    def top_feature_importances(self, top_n: int = 5) -> dict:
        cb = self.models["CB_deep"]
        names = self.pipeline.feature_columns
        importances = cb.get_feature_importance()
        order = np.argsort(importances)[::-1][:top_n]
        return {str(names[i]): float(importances[i]) for i in order}


def run_scoring(raw_df: pd.DataFrame):
    engine = ScoringEngine().load()
    views = engine.pipeline.transform(raw_df)
    preds = engine.predict(views)
    importances = engine.top_feature_importances(5)
    return preds, importances


if __name__ == "__main__":
    from .load_data import load_input

    df = load_input()
    predictions, fi = run_scoring(df)
    print("Топ-5 важностей признаков:")
    print(json.dumps(fi, indent=2))
