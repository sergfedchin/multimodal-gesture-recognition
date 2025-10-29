# multimodal-gesture-recognition

Как запустить:
0. Склонировать репозиторий yolov13:
```sh
git clone https://github.com/iMoonLab/yolov13.git
```
1. Установить зависимости через `uv`
2. Положить папку с изображениями жеста в `data/gestures` (например, `data/gestures/call/*.png`)
3. Положить аннотации в `data/annotations/test|train|dev/*.json`
4. Положить веса моделей в `checkpoints/` (`checkpoints/ppd.pth` и `checkpoints/depth_anything_v2_vitl.pth`)
5. Обновить название жеста и проврить пути в `config.toml`
6. Запустить `python process_gesture.py`