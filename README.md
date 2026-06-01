# ML Scoring Service

Коротко: это Docker-сервис для inference модели из соревнования ML 2025.

Берет файл `test.csv` из папки `input/` и пишет результаты в `output/`.
Работает только на CPU.

## Что на выходе

После запуска появятся 3 файла:
- `output/sample_submission.csv`
- `output/feature_importances.json` (топ-5 важностей признаков)
- `output/prediction_density.png` (график плотности предсказаний)

## Быстрый запуск (рекомендуется)

1. Положите входной файл:

```bash
cp /путь/к/вашему/test.csv input/test.csv
```

2. Соберите образ:

```bash
docker compose build
```

3. Запустите сервис:

```bash
docker compose run --rm scoring
```

4. Проверьте результат:

```bash
ls -l output/
```

## Запуск без compose

```bash
docker build -t ml-scoring-service:latest .
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  ml-scoring-service:latest
```

## Где что находится

- `src/load_data.py` — загрузка входного файла  
- `src/preprocess.py` — препроцессинг  
- `src/score.py` — скоринг модели  
- `src/export_results.py` — выгрузка результатов  
- `src/main.py` — запуск всего пайплайна  
- `models/` — готовые артефакты моделей  

## Частые проблемы

- `input/` пустая: положите `test.csv` в `input/`.
- Нет файлов в `output/`: проверьте монтирование `./output:/app/output`.
- Ошибка по колонкам: используйте `test.csv` в формате соревнования.
