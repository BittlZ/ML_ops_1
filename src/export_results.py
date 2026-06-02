from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt

from . import config


def write_submission(preds: np.ndarray) -> str:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / config.SUBMISSION_FILENAME
    sub = pd.DataFrame({"index": range(len(preds)),
                        "prediction": np.clip(preds, config.TARGET_MIN, config.TARGET_MAX)})
    sub.to_csv(path, index=False)
    print(f"[выгрузка] Файл сабмита сохранен: {path} ({len(sub)} строк)")
    return str(path)


def write_feature_importances(importances: dict) -> str:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / config.FEATURE_IMPORTANCE_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2, ensure_ascii=False)
    print(f"[выгрузка] Файл важностей признаков сохранен: {path}")
    return str(path)


def write_density_plot(preds: np.ndarray) -> str:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / config.DENSITY_PLOT_FILENAME

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(preds, bins=60, density=True, color="#3498db",
            alpha=0.55, edgecolor="black", label="гистограмма")

    try:
        from scipy.stats import gaussian_kde
        if preds.std() > 1e-6:
            kde = gaussian_kde(preds)
            xs = np.linspace(preds.min(), preds.max(), 400)
            ax.plot(xs, kde(xs), color="#e74c3c", lw=2, label="оценка плотности (KDE)")
    except Exception:
        pass

    ax.set_title("Плотность распределения предсказаний")
    ax.set_xlabel("Предсказанное значение target")
    ax.set_ylabel("Плотность")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[выгрузка] График плотности сохранен: {path}")
    return str(path)


def export_all(preds: np.ndarray, importances: dict) -> dict:
    return {
        "сабмит": write_submission(preds),
        "важности_признаков": write_feature_importances(importances),
        "график_плотности": write_density_plot(preds),
    }
