#!/bin/bash
# Скрипт для скачивания всех данных для проекта обнаружения генеративного ИИ

# Определяем базовую директорию проекта
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"

echo "Установка необходимых зависимостей..."
pip install gdown pandas numpy matplotlib

echo -e "\n=== Скачивание данных для задачи 1 (бинарная классификация) ==="
python "${SCRIPTS_DIR}/download_task1_data.py"

echo -e "\n=== Скачивание данных для задачи 2 (многоклассовая классификация) ==="
python "${SCRIPTS_DIR}/download_task2_data.py"

echo -e "\nВсе данные успешно скачаны и подготовлены."
echo "Структура директорий:"
find "${BASE_DIR}/data" -type f | sort

echo -e "\nСледующие шаги:"
echo "1. Запустите скрипты анализа данных для изучения распределений"
echo "2. Разработайте модели для задач классификации"
echo "3. Обучите модели на тренировочных данных"
echo "4. Оцените качество на валидационной выборке"
