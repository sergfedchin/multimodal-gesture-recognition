# Препроцессинг корпуса HaGRIDv2.1 

Скрипт извлекает карты глубины вырезанных тела и рук людей.

## Как запустить

1. Склонировать репозиторий `yolov13`:
    ```sh
    git clone https://github.com/iMoonLab/yolov13.git
    ```
2. Установить зависимости через `uv`
3. Положить папку с изображениями жеста в `data/gestures` (например, `data/gestures/call/*.png`)
4. Положить аннотации в `data/annotations/test|train|dev/*.json`
5. Положить веса моделей в `checkpoints/` (`checkpoints/ppd.pth` и `checkpoints/depth_anything_v2_vitl.pth`)
6. Обновить название жеста и проврить пути в `config.toml`
7. Запустить скрипт:
    ```sh
    .venv/bin/python process_gesture.py
    ```