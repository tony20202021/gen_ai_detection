#!/usr/bin/env python3
"""
Скрипт для генерации предсказаний и файлов для отправки
"""

import os
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import joblib
from datetime import datetime
import json
from tqdm import tqdm
from glob import glob
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# Создаем директорию для логов, если её нет
os.makedirs("../logs", exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/generate_submissions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Генерация предсказаний для отправки')
    parser.add_argument('--task', type=int, choices=[1, 2], required=True, 
                        help='Номер задачи (1 - бинарная классификация, 2 - многоклассовая)')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['logistic', 'svm', 'rf', 'xgb', 'mlp', 'all'],
                        help='Тип модели для использования (если не указано, будут использованы все доступные модели)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Директория для сохранения результатов (по умолчанию: ../results/submissions)')
    parser.add_argument('--version', type=str, default='v01',
                        help='Версия отправки (например, v01)')
    parser.add_argument('--force', action='store_true',
                        help='Принудительно пересоздать предсказания, даже если они уже существуют')
    
    args = parser.parse_args()
    
    # Установка значения по умолчанию для output_dir, если не указано
    if args.output_dir is None:
        args.output_dir = os.path.join("..", "results", "submissions")
        
    return args

def get_data_paths(task):
    """Получение путей к тренировочным и валидационным данным"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(base_dir, "data")
    
    if task == 1:
        train_path = os.path.join(data_dir, 'task1', 'train.jsonl')
        val_path = os.path.join(data_dir, 'task1', 'val.jsonl')
    else:  # task == 2
        train_path = os.path.join(data_dir, 'task2', 'subtask2_train.jsonl')
        val_path = os.path.join(data_dir, 'task2', 'subtask2_dev.jsonl')
    
    return train_path, val_path

def get_model_paths(task, model_type=None):
    """Поиск всех доступных моделей для указанной задачи и типа модели"""
    models_dir = os.path.join("..", "results", "models")
    
    if model_type and model_type != 'all':
        # Если указан тип модели, ищем только модели этого типа
        model_pattern = os.path.join(models_dir, f"task{task}_{model_type}_model_*.joblib")
    else:
        # Иначе ищем все модели для данной задачи
        model_pattern = os.path.join(models_dir, f"task{task}_*_model_*.joblib")
    
    model_paths = glob(model_pattern)
    
    if not model_paths:
        logger.error(f"Не найдено моделей для задачи {task}" + 
                     (f" и типа {model_type}" if model_type and model_type != 'all' else ""))
        return []
    
    logger.info(f"Найдено {len(model_paths)} моделей для задачи {task}" + 
                (f" и типа {model_type}" if model_type and model_type != 'all' else ""))
    for path in model_paths:
        logger.info(f"  - {os.path.basename(path)}")
    
    return model_paths

def load_model(model_path):
    """Загрузка обученной модели из файла"""
    logger.info(f"Загрузка модели из {model_path}")
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Ошибка при загрузке модели: {e}")
        return None

def load_data(data_path):
    """Загрузка данных из файла"""
    logger.info(f"Загрузка данных из {data_path}")
    try:
        df = pd.read_json(data_path, lines=True)
        logger.info(f"Загружено {len(df)} примеров")
        return df
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return None

def load_embeddings(task, subset='all'):
    """Загрузка уже извлеченных эмбедингов для заданной задачи"""
    embeddings_dir = os.path.join("..", "results", "embeddings")
    
    # Ищем файлы эмбедингов с пониженной размерностью
    pattern = os.path.join(embeddings_dir, f"task{task}_embeddings_*_with_dim_reduction.pkl")
    embeddings_paths = glob(pattern)
    
    if not embeddings_paths:
        logger.error(f"Не найдены файлы эмбедингов для задачи {task}")
        return None, None, None
    
    # Берем первый найденный файл эмбедингов
    embeddings_path = embeddings_paths[0]
    logger.info(f"Загрузка эмбедингов из {embeddings_path}")
    
    try:
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = data.get("embeddings")
        embeddings_pca = data.get("embeddings_pca")
        metadata = data.get("metadata")
        
        # Проверяем, что эмбединги и метаданные были загружены успешно
        if embeddings is None or metadata is None:
            logger.error("Файл эмбедингов не содержит необходимых данных")
            return None, None, None
        
        # Фильтруем эмбединги и метаданные в зависимости от подмножества
        if subset != 'all':
            if 'split' in metadata.columns:
                if subset == 'train':
                    mask = metadata['split'] == 'train'
                elif subset == 'val':
                    mask = metadata['split'] == 'val'
                else:
                    mask = slice(None)  # все данные
                
                embeddings = embeddings[mask]
                if embeddings_pca is not None:
                    embeddings_pca = embeddings_pca[mask]
                metadata = metadata[mask]
                
                logger.info(f"Выбрано подмножество '{subset}': {len(metadata)} примеров")
        
        logger.info(f"Загружено {len(metadata)} эмбедингов размерности {embeddings.shape[1]}")
        if embeddings_pca is not None:
            logger.info(f"Доступны PCA-эмбединги размерности {embeddings_pca.shape[1]}")
        
        return embeddings, embeddings_pca, metadata
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке эмбедингов: {e}")
        return None, None, None

def apply_model(model, embeddings):
    """Применение модели к эмбедингам для получения предсказаний"""
    logger.info("Генерация предсказаний")
    try:
        predictions = model.predict(embeddings)
        
        # Если модель возвращает вероятности, берем их тоже
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(embeddings)
            logger.info(f"Получены вероятности с формой {probabilities.shape}")
        else:
            probabilities = None
        
        logger.info(f"Сгенерировано {len(predictions)} предсказаний")
        logger.info(f"Распределение классов в предсказаниях: {np.unique(predictions, return_counts=True)}")
        
        return predictions, probabilities
    except Exception as e:
        logger.error(f"Ошибка при генерации предсказаний: {e}")
        raise e

def generate_submission_file(df, predictions, task, output_dir, model_name, version):
    """Генерация файла с предсказаниями для отправки"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Формирование имени файла
    today = datetime.now().strftime("%Y_%m_%d")
    filename = f"task_{task}_anton_mikhalev_{today}_{version}_{model_name}.jsonl"
    output_path = os.path.join(output_dir, filename)
    
    # Проверка, существует ли файл уже
    if os.path.exists(output_path):
        logger.warning(f"Файл предсказаний уже существует: {output_path}")
        return output_path
    
    # Если поля 'id' нет, используем индекс строки (начиная с 0)
    if 'id' not in df.columns:
        logger.warning(f"В загруженных данных отсутствует поле 'id'. Используем порядковый номер записи.")
        ids = [str(i) for i in range(len(df))]
    else:
        ids = df['id'].astype(str).tolist()
    
    # Проверяем, что количество предсказаний соответствует количеству записей
    if len(ids) != len(predictions):
        logger.error(f"Несоответствие размеров: {len(ids)} записей и {len(predictions)} предсказаний")
        # Усекаем до минимального размера, чтобы продолжить работу
        min_len = min(len(ids), len(predictions))
        ids = ids[:min_len]
        predictions = predictions[:min_len]
    
    # Запись в JSONL формат
    with open(output_path, 'w') as f:
        for i, pred in enumerate(predictions):
            f.write(json.dumps({'id': ids[i], 'label': int(pred)}) + '\n')
    
    logger.info(f"Файл с предсказаниями сохранен в {output_path}")
    return output_path

def load_label_encoder(task):
    """Загрузка LabelEncoder для задачи (если применимо)"""
    encoder_path = os.path.join("..", "results", "encoders", f"task{task}_label_encoder.pkl")
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"Загружен LabelEncoder для задачи {task}")
        return label_encoder
    return None

def main():
    args = parse_args()
    
    # Создаем директорию для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Получаем пути к данным
    train_path, val_path = get_data_paths(args.task)
    
    # Загружаем тренировочные и валидационные данные
    train_df = load_data(train_path)
    val_df = load_data(val_path)
    
    if train_df is None or val_df is None:
        logger.error("Не удалось загрузить данные")
        return
    
    # Загружаем уже извлеченные эмбединги
    _, train_embeddings_pca, train_metadata = load_embeddings(args.task, subset='train')
    _, val_embeddings_pca, val_metadata = load_embeddings(args.task, subset='val')
    
    if train_embeddings_pca is None or val_embeddings_pca is None:
        logger.error("Не удалось загрузить эмбединги")
        return
    
    # Получаем пути к моделям
    model_paths = get_model_paths(args.task, args.model_type)
    
    if not model_paths:
        return
    
    # Загружаем LabelEncoder, если есть
    label_encoder = load_label_encoder(args.task)
    
    # Для каждой модели генерируем предсказания и файлы для отправки
    for model_path in model_paths:
        # Извлекаем имя модели из пути
        model_basename = os.path.basename(model_path)
        model_name = model_basename.split('_model_')[0].split(f'task{args.task}_')[1]
        logger.info(f"Обработка модели: {model_name}")
        
        # Проверяем, существуют ли уже файлы предсказаний для этой модели
        today = datetime.now().strftime("%Y_%m_%d")
        train_output_path = os.path.join(args.output_dir, f"task_{args.task}_anton_mikhalev_{today}_{args.version}_{model_name}_train.jsonl")
        val_output_path = os.path.join(args.output_dir, f"task_{args.task}_anton_mikhalev_{today}_{args.version}_{model_name}_val.jsonl")
        
        if os.path.exists(train_output_path) and os.path.exists(val_output_path) and not args.force:
            logger.info(f"Пропуск модели {model_name}, файлы предсказаний уже существуют")
            continue
        
        # Загружаем модель
        model = load_model(model_path)
        if model is None:
            continue
        
        try:
            # Генерируем предсказания для тренировочных данных
            logger.info(f"Генерация предсказаний для тренировочных данных")
            train_predictions, train_probabilities = apply_model(model, train_embeddings_pca)
            
            # Генерируем предсказания для валидационных данных
            logger.info(f"Генерация предсказаний для валидационных данных")
            val_predictions, val_probabilities = apply_model(model, val_embeddings_pca)
            
            # Проверяем, нужно ли использовать label_encoder для задачи 2
            if args.task == 2 and label_encoder is not None:
                logger.info("Применение маппинга меток для задачи 2")
                # Если у нас числовые предсказания, убеждаемся что они соответствуют правильным классам
                class_mapping = {i: i for i in range(6)}  # По умолчанию отображаем в себя
                try:
                    # Пытаемся найти соответствие между предсказанными числами и нужными классами
                    inverse_mapping = {label: i for i, label in enumerate(label_encoder.classes_)}
                    logger.info(f"Маппинг меток: {inverse_mapping}")
                    
                    # Применяем маппинг к предсказаниям
                    if len(np.unique(train_predictions)) <= len(inverse_mapping):
                        train_predictions = np.array([inverse_mapping.get(label, label) for label in train_predictions])
                    if len(np.unique(val_predictions)) <= len(inverse_mapping):
                        val_predictions = np.array([inverse_mapping.get(label, label) for label in val_predictions])
                    
                    logger.info(f"Распределение классов после маппинга (тренировочные): {np.unique(train_predictions, return_counts=True)}")
                    logger.info(f"Распределение классов после маппинга (валидационные): {np.unique(val_predictions, return_counts=True)}")
                    
                except Exception as e:
                    logger.warning(f"Не удалось применить обратное преобразование LabelEncoder: {e}")
            
            # Генерируем файлы предсказаний
            try:
                train_output = generate_submission_file(train_df, train_predictions, args.task, 
                                                        args.output_dir, f"{model_name}_train", args.version)
                val_output = generate_submission_file(val_df, val_predictions, args.task, 
                                                      args.output_dir, f"{model_name}_val", args.version)
                
                # Сохраняем вероятности для дальнейшего анализа, если они доступны
                if train_probabilities is not None:
                    train_probs_path = train_output.replace('.jsonl', '_probabilities.npy')
                    np.save(train_probs_path, train_probabilities)
                    logger.info(f"Вероятности для тренировочных данных сохранены в {train_probs_path}")
                
                if val_probabilities is not None:
                    val_probs_path = val_output.replace('.jsonl', '_probabilities.npy')
                    np.save(val_probs_path, val_probabilities)
                    logger.info(f"Вероятности для валидационных данных сохранены в {val_probs_path}")
            except Exception as e:
                logger.error(f"Ошибка при сохранении предсказаний для модели {model_name}: {e}")
                # Выводим более подробную информацию для отладки
                logger.error(f"Типы данных: train_df={type(train_df)}, train_predictions={type(train_predictions)}")
                logger.error(f"Размеры данных: train_df={len(train_df)}, train_predictions={len(train_predictions)}")
                if 'id' in train_df.columns:
                    logger.error(f"Пример ID: {train_df['id'].iloc[0]}, тип: {type(train_df['id'].iloc[0])}")
                    logger.error(f"Пример предсказания: {train_predictions[0]}, тип: {type(train_predictions[0])}")
                continue
            
        except Exception as e:
            logger.error(f"Ошибка при обработке модели {model_name}: {e}")
            continue
    
    logger.info("Генерация предсказаний завершена успешно.")

if __name__ == "__main__":
    main()
