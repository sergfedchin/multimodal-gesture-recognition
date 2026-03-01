# Демо-приложение мультимодальной системы распознавания жестов

Этот репозиторий подготовлен для запуска локально и публикации в **Hugging Face Spaces**.

## Что изменено для Spaces

- Демо запускается **без attention visualization** (ускоряет и упрощает CPU-режим).
- Чекпоинты автоматически подгружаются из HF Hub: **`sergfedchin/gesture-checkpoints`** в папку `checkpoints/`.
- Локальный пакет `yolov13` ставится из подпапки проекта через `requirements.txt` (без `git clone`).

---

## Структура, которая ожидается в репозитории

```text
.
├── app.py
├── requirements.txt
├── checkpoints/
│   └── .gitkeep
├── preprocess/
│   └── yolov13/   # локальная копия yolov13
└── ...
```

---

## Локальный запуск (CPU)

1. Установить зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Запустить:

   ```bash
   python app.py
   ```

При первом запуске приложение скачает чекпоинты из `sergfedchin/gesture-checkpoints` в `checkpoints/`.

---
