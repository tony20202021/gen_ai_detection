# Структура директорий проекта по обнаружению генеративного ИИ

Ниже приведена структура директорий и файлов проекта с описанием их назначения:

```
/workspace-SR006.nfs2/amikhalev/repos/gen_ai_detection/
├── data/                       # Директория для хранения данных
│   ├── task1/                  # Данные для задачи 1 (бинарная классификация)
│   └── task2/                  # Данные для задачи 2 (многоклассовая классификация)
├── docs/                       # Документация проекта
│   └── project_description.md  # Описание проекта и его задач
├── model/                      # Основные скрипты для работы с моделями
│   ├── clustering.py           # Скрипт для кластеризации эмбедингов
│   ├── extract_embeddings.py   # Скрипт для извлечения эмбедингов из текстов
│   ├── run_pipeline.bat        # Скрипт запуска пайплайна для Windows
│   ├── run_pipeline.sh         # Скрипт запуска пайплайна для Linux/macOS
│   ├── train_models.py         # Скрипт для обучения и оценки моделей
│   └── visualize_embeddings.ipynb  # Ноутбук для визуализации эмбедингов и результатов
├── notebooks/                  # Jupyter ноутбуки для анализа и экспериментов
│   ├── EDA_PAN.ipynb           # Ноутбук с разведочным анализом данных
│   └── 'EDA_PAN - cleared output.ipynb'  # Версия ноутбука с очищенными выводами
├── scripts/                    # Вспомогательные скрипты
│   ├── download_all_data.sh    # Скрипт для скачивания всех данных
│   ├── download_task1_data.py  # Скрипт для скачивания данных задачи 1
│   ├── download_task2_data.py  # Скрипт для скачивания данных задачи 2
│   ├── setup_environment.bat   # Скрипт настройки окружения для Windows
│   └── setup_environment.sh    # Скрипт настройки окружения для Linux/macOS
├── environment.yml             # Файл с конфигурацией conda-окружения
├── README.md                   # Общее описание проекта и инструкции
└── requirements.txt            # Список зависимостей Python
```

## Дополнительные директории (создаются во время выполнения)

При выполнении скриптов будут созданы следующие директории:

```
/workspace-SR006.nfs2/amikhalev/repos/gen_ai_detection/
├── data/
│   ├── embeddings/             # Сохраненные эмбединги текстов
│   │   ├── task1_embeddings_*.pkl
│   │   └── task2_embeddings_*.pkl
│   └── clustering/             # Результаты кластеризации
│       ├── task1_*_clustering_results.pkl
│       └── task2_*_clustering_results.pkl
├── models/                     # Директория для сохранения обученных моделей
│   ├── task1_*_model_*.joblib
│   └── task2_*_model_*.joblib
└── results/                    # Директория для сохранения результатов и метрик
    ├── task1_*_metrics_*.json
    └── task2_*_accuracy_comparison_*.png
```

## Рабочий процесс

1. **Установка и настройка окружения**:
   - Используйте `scripts/setup_environment.sh` (Linux/macOS) или `scripts/setup_environment.bat` (Windows)

2. **Скачивание данных**:
   - Используйте `scripts/download_task1_data.py` и `scripts/download_task2_data.py` или
   - `scripts/download_all_data.sh` для скачивания всех данных

3. **Разведочный анализ данных**:
   - Изучите данные с помощью ноутбуков в директории `notebooks/`

4. **Выполнение пайплайна обработки**:
   - Используйте `model/run_pipeline.sh` (Linux/macOS) или `model/run_pipeline.bat` (Windows) для запуска полного пайплайна
   - Или выполните отдельные шаги с помощью скриптов в директории `model/`

5. **Анализ результатов**:
   - Используйте `model/visualize_embeddings.ipynb` для исследования полученных эмбедингов и результатов кластеризации
   - Изучите метрики моделей в директории `results/`