---
title: Gestures Demo
emoji: 👋
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.4.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

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

## Пошагово: публикация в Hugging Face Space (CPU / бесплатно)

### 1) Создать Space

1. Перейди на https://huggingface.co/new-space
2. Выбери:
   - **SDK:** Gradio
   - **Visibility:** Public
   - **Hardware:** CPU Basic (бесплатно)

### 2) Подготовить файлы

В корне Space должны быть минимум:
- `app.py`
- `requirements.txt`
- код проекта (`multimodal_gesture_recognition`, `preprocess`, и т.д.)

### 3) Залить код

```bash
git clone https://huggingface.co/spaces/<username>/<space_name>
cd <space_name>
# скопировать сюда файлы этого проекта

git add .
git commit -m "Initial CPU Space setup"
git push
```

### 4) Дождаться сборки

- Открой вкладку **Build logs** в Space.
- Убедись, что `-e ./preprocess/yolov13` установился успешно.
- При первом запуске будут скачаны веса из `sergfedchin/gesture-checkpoints`.

### 5) Проверить запуск

- Открой UI Space.
- Загрузи изображение из примеров.
- Проверь, что есть детекции, глубина и вероятности классов.

---

## Нужна ли оплата

- **CPU Space**: обычно доступен бесплатно.
- **GPU Space**: как правило, платный (стоимость зависит от типа GPU и тарифа).

Если позже потребуется GPU:
1. Settings → Hardware → выбрать GPU.
2. Перезапустить Space после смены железа.

---

## Советы по стабильности на CPU

- Первую загрузку делать терпеливо: скачивание весов + холодный старт.
- Если сборка падает на тяжелых зависимостях, зафиксировать версии `torch/torchvision` под конкретный рантайм Spaces.
- Держать веса в отдельном model repo (как сейчас), а не в кодовом репозитории Space.
