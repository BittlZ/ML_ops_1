from __future__ import annotations

import sys
import time

from . import config
from .load_data import load_input
from .score import ScoringEngine
from .export_results import export_all


def main() -> int:
    t0 = time.time()
    print("=" * 60)
    print("Сервис ML-инференса — одиночный запуск")
    print(f"  Входная директория (INPUT_DIR): {config.INPUT_DIR}")
    print(f"  Выходная директория (OUTPUT_DIR): {config.OUTPUT_DIR}")
    print(f"  Директория моделей (MODELS_DIR): {config.MODELS_DIR}")
    print("=" * 60)
    raw_df = load_input()

    engine = ScoringEngine().load()
    print("[главный] Выполняется препроцессинг входного файла...")
    views = engine.pipeline.transform(raw_df)
    print("[главный] Выполняется скоринг...")
    preds = engine.predict(views)
    importances = engine.top_feature_importances(5)

    outputs = export_all(preds, importances)

    print("=" * 60)
    print(f"Готово за {time.time() - t0:.1f}с. Выходные файлы:")
    for name, path in outputs.items():
        print(f"  - {name}: {path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
