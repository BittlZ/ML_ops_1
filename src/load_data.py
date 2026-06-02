from __future__ import annotations
from pathlib import Path
import pandas as pd
from . import config


def find_input_file() -> Path:
    expected = config.INPUT_DIR / config.INPUT_FILENAME
    if expected.exists():
        return expected

    if not config.INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Входная директория '{config.INPUT_DIR}' не существует. "
            f"Смонтируйте данные через -v ./input:/app/input и поместите "
            f"файл '{config.INPUT_FILENAME}' внутрь."
        )

    csvs = sorted(config.INPUT_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"В директории '{config.INPUT_DIR}' не найден CSV-файл. "
            f"Ожидался файл '{config.INPUT_FILENAME}'."
        )
    return csvs[0]


def load_input() -> pd.DataFrame:
    path = find_input_file()
    print(f"[загрузка] Чтение входного файла: {path}")
    df = pd.read_csv(path)
    print(f"[загрузка] Загружено {len(df)} строк, {df.shape[1]} столбцов.")

    missing = [c for c in config.RAW_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[загрузка] Во входном файле отсутствуют обязательные столбцы: {missing}. "
            f"Ожидаемые столбцы: {config.RAW_FEATURE_COLS}"
        )

    df = df[config.RAW_FEATURE_COLS].copy()
    return df


if __name__ == "__main__":
    frame = load_input()
    print(frame.head())
