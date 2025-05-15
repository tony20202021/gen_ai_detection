#!/usr/bin/env python3
"""
Скрипт для скачивания данных для задачи 1 (бинарная классификация) проекта обнаружения генеративного ИИ
"""

import os
import subprocess
import zipfile
import pandas as pd

# Создаем директорию для данных, если ее нет
os.makedirs("data/task1", exist_ok=True)

def download_with_gdown(file_id, output_file):
    """Скачивание файла с Google Drive с помощью gdown."""
    cmd = ["gdown", file_id, "-O", output_file]
    print(f"Выполняется команда: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def unzip_file(zip_file, extract_to="."):
    """Распаковка zip архива."""
    print(f"Распаковка {zip_file} в {extract_to}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Распаковка {zip_file} завершена.")

def main():
    # Изменяем рабочую директорию на директорию с данными
    os.chdir("data/task1")
    
    # Скачиваем файлы данных
    print("Скачивание файлов данных...")
    download_with_gdown("1lGIpg3OhOAlNPgBDTAv5AYgpIdEz2Isa", "14962653.zip")
    download_with_gdown("1_R7mVJMgVxdlC5-TjdLnxB8HTX9unZ-_", "pan25-generative-ai-detection-task1-train.zip")
    
    # Распаковываем архивы
    unzip_file("14962653.zip")
    unzip_file("pan25-generative-ai-detection-task1-train.zip")
    
    # Проверяем, что файлы данных существуют
    for file in ["train.jsonl", "val.jsonl"]:
        if os.path.exists(file):
            print(f"Файл {file} успешно скачан и распакован.")
            # Проверим количество записей в файле
            try:
                df = pd.read_json(file, lines=True)
                print(f"  - Количество записей в {file}: {len(df)}")
            except Exception as e:
                print(f"  - Ошибка при чтении {file}: {e}")
        else:
            print(f"Внимание: файл {file} не найден после распаковки!")
    
    print("Скачивание и распаковка данных для задачи 1 завершены.")
    os.chdir("../..")  # Возвращаемся в исходную директорию

if __name__ == "__main__":
    main()
    