# Демо-приложение мультимодальной системы распознавания жестов

## Для запуска:

1. Скачать все веса моделей в папку `checkpoints/`
2. Установить зависимости
  ```shell
  git clone git@github.com:iMoonLab/yolov13.git preprocess/yolov13
  uv sync --frozen
  uv pip install -e preprocess/yolov13
  ```
3. Запустить
  ```shell
  uv run app.py
  ```