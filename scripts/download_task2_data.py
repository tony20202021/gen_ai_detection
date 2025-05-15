#!/usr/bin/env python3
"""
Скрипт для скачивания данных для задачи 2 (многоклассовая классификация) проекта обнаружения генеративного ИИ
"""

import os
import subprocess
import pandas as pd

# Создаем директорию для данных, если ее нет
os.makedirs("data/task2", exist_ok=True)

def download_with_gdown(file_id, output_file):
    """Скачивание файла с Google Drive с помощью gdown."""
    cmd = ["gdown", file_id, "-O", output_file]
    print(f"Выполняется команда: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    # Изменяем рабочую директорию на директорию с данными
    os.chdir("data/task2")
    
    # Скачиваем файлы данных
    print("Скачивание файлов данных...")
    download_with_gdown("1rNQTkhkVG9nzcT97Nk_WyJd80ZaacT0-", "subtask2_dev.jsonl")
    download_with_gdown("1u5C4o_fmjL5nQ_RtgLDShuG97Ix6_KGK", "subtask2_train.jsonl")
    
    # Проверяем, что файлы данных существуют и выводим статистику
    for file in ["subtask2_train.jsonl", "subtask2_dev.jsonl"]:
        if os.path.exists(file):
            print(f"Файл {file} успешно скачан.")
            # Проверим количество записей в файле и распределение классов
            try:
                df = pd.read_json(file, lines=True)
                print(f"  - Количество записей в {file}: {len(df)}")
                if 'label_text' in df.columns:
                    class_dist = df['label_text'].value_counts()
                    print(f"  - Распределение классов:")
                    for cls, count in class_dist.items():
                        print(f"    * {cls}: {count} ({count/len(df)*100:.1f}%)")
            except Exception as e:
                print(f"  - Ошибка при чтении {file}: {e}")
        else:
            print(f"Внимание: файл {file} не найден!")
    
    print("Скачивание данных для задачи 2 завершено.")
    os.chdir("../..")  # Возвращаемся в исходную директорию

if __name__ == "__main__":
    main()
    