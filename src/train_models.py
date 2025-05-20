#!/usr/bin/env python3
"""
Скрипт для обучения и оценки моделей классификации на основе эмбедингов
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import pickle
import argparse
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import joblib

# Создаем директорию для логов, если её нет
os.makedirs("../logs", exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/train_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение и оценка моделей классификации')
    parser.add_argument('--task', type=int, choices=[1, 2], required=True, 
                        help='Номер задачи (1 - бинарная классификация, 2 - многоклассовая)')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Путь к файлу с эмбедингами')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Директория для сохранения обученных моделей (по умолчанию: ../results/models)')
    parser.add_argument('--use_pca', action='store_true',
                        help='Использовать PCA-эмбединги, если доступны')
    parser.add_argument('--cv', type=int, default=5,
                        help='Количество фолдов для кросс-валидации')
    parser.add_argument('--model_type', type=str, default='all',
                        choices=['logistic', 'svm', 'rf', 'xgb', 'mlp', 'all'],
                        help='Тип модели для обучения')
    parser.add_argument('--grid_search', action='store_true',
                        help='Выполнить поиск гиперпараметров с помощью GridSearchCV')
    parser.add_argument('--balance', action='store_true',
                        help='Учитывать несбалансированность классов')
    
    args = parser.parse_args()
    
    # Установка значения по умолчанию для output_dir, если не указано
    if args.output_dir is None:
        args.output_dir = os.path.join("..", "results", "models")
        
    return args

def load_embeddings(embeddings_path):
    """Загрузка эмбедингов из файла"""
    logger.info(f"Загрузка эмбедингов из {embeddings_path}")
    try:
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            embeddings = data.get("embeddings")
            metadata = data.get("metadata")
            embeddings_pca = data.get("embeddings_pca")
        else:
            embeddings = data
            metadata = None
            embeddings_pca = None
            
        logger.info(f"Загружено {embeddings.shape[0]} эмбедингов размерности {embeddings.shape[1]}")
        return embeddings, metadata, embeddings_pca
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке эмбедингов: {e}")
        return None, None, None

def get_label_column(task, metadata):
    """Определение колонки с метками в зависимости от задачи"""
    if task == 1:
        label_column = 'label'
    else:  # task == 2
        label_column = 'label_text' if 'label_text' in metadata.columns else 'label'
    
    return label_column

def get_models(task, balance=False):
    """Получение набора моделей для обучения"""
    if task == 1:
        # Для бинарной классификации
        if balance:
            class_weight = 'balanced'
        else:
            class_weight = None
        
        models = {
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=42))
            ]),
            'svm': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(probability=True, class_weight=class_weight, random_state=42))
            ]),
            'rf': Pipeline([
                ('classifier', RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=42))
            ]),
            'xgb': Pipeline([
                ('classifier', XGBClassifier(
                    objective='binary:logistic', 
                    n_estimators=100, 
                    scale_pos_weight=1 if not balance else None,
                    random_state=42
                ))
            ]),
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(max_iter=500, random_state=42))
            ])
        }
    else:
        # Для многоклассовой классификации
        if balance:
            class_weight = 'balanced'
        else:
            class_weight = None
        
        models = {
            'logistic': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, multi_class='multinomial', 
                                                class_weight=class_weight, random_state=42))
            ]),
            # 'svm': Pipeline([
            #     ('scaler', StandardScaler()),
            #     ('classifier', SVC(probability=True, class_weight=class_weight, random_state=42))
            # ]),
            'rf': Pipeline([
                ('classifier', RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=42))
            ]),
            'xgb': Pipeline([
                ('classifier', XGBClassifier(objective='multi:softprob', n_estimators=100, random_state=42))
            ]),
            'mlp': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', MLPClassifier(max_iter=500, random_state=42))
            ])
        }
    
    return models

def get_param_grids():
    """Получение сеток параметров для поиска гиперпараметров"""
    param_grids = {
        'logistic': {
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0],
            'classifier__solver': ['liblinear', 'saga']
        },
        'svm': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'rf': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'xgb': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 6, 9],
            'classifier__learning_rate': [0.01, 0.1, 0.3]
        },
        'mlp': {
            'classifier__hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25)],
            'classifier__alpha': [0.0001, 0.001, 0.01],
            'classifier__learning_rate': ['constant', 'adaptive']
        }
    }
    
    return param_grids

def train_and_evaluate(X_train, y_train, X_val, y_val, models, model_type, grid_search=False):
    """Обучение и оценка моделей"""
    results = {}
    
    if model_type != 'all':
        if model_type in models:
            models_to_train = {model_type: models[model_type]}
        else:
            logger.error(f"Неизвестный тип модели: {model_type}")
            return None
    else:
        models_to_train = models
    
    param_grids = get_param_grids() if grid_search else None
    
    for name, model in models_to_train.items():
        logger.info(f"Обучение модели: {name}")
        
        if grid_search:
            logger.info(f"  Выполняется поиск гиперпараметров...")
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logger.info(f"  Лучшие параметры: {grid.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train, y_train)
        
        # Оценка на валидационной выборке
        y_pred = best_model.predict(X_val)
        
        # Вычисление метрик
        accuracy = accuracy_score(y_val, y_pred)
        
        try:
            # Для бинарной классификации вычисляем ROC-AUC
            if len(np.unique(y_train)) == 2:
                y_prob = best_model.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_prob)
            else:
                # Для многоклассовой используем многоклассовый ROC-AUC
                y_prob = best_model.predict_proba(X_val)
                roc_auc = roc_auc_score(y_val, y_prob, multi_class='ovr')
        except Exception as e:
            logger.error(f"Ошибка при вычислении ROC-AUC: {e}")
            roc_auc = None
        
        # Формирование отчета о классификации
        report = classification_report(y_val, y_pred, output_dict=True)
        
        # Сохранение результатов
        results[name] = {
            'model': best_model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        logger.info(f"  Accuracy: {accuracy:.4f}")
        if roc_auc is not None:
            logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    
    return results

def save_results(results, output_dir, task):
    """Сохранение результатов обучения и моделей"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем модели
    for name, result in results.items():
        model_path = os.path.join(output_dir, f"task{task}_{name}_model_{timestamp}.joblib")
        joblib.dump(result['model'], model_path)
        logger.info(f"Модель {name} сохранена в {model_path}")
    
    # Сохраняем метрики и отчеты
    metrics = {}
    for name, result in results.items():
        metrics[name] = {
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc'],
            'classification_report': result['classification_report'],
            'confusion_matrix': result['confusion_matrix']
        }
    
    metrics_path = os.path.join(output_dir, f"task{task}_metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Метрики сохранены в {metrics_path}")
    
    return metrics_path

def plot_results(results, output_dir, task):
    """Построение графиков с результатами"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Определим директорию для результатов
    results_dir = os.path.join("..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сравнение accuracy разных моделей
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    plt.bar(model_names, accuracies, color=sns.color_palette('viridis', len(model_names)))
    plt.title(f'Сравнение точности моделей (Задача {task})', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"task{task}_accuracy_comparison_{timestamp}.png"))
    
    # Построение матриц ошибок для лучшей модели
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model_results = results[best_model_name]
    
    plt.figure(figsize=(8, 6))
    cm = np.array(best_model_results['confusion_matrix'])
    
    # Нормализация матрицы ошибок по строкам
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Матрица ошибок для {best_model_name} (Задача {task})', fontsize=14)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"task{task}_{best_model_name}_confusion_matrix_{timestamp}.png"))
    
    logger.info(f"Графики результатов сохранены в {results_dir}")

def main():
    args = parse_args()
    
    # Создаем директорию для сохранения моделей
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем эмбединги
    embeddings, metadata, embeddings_pca = load_embeddings(args.embeddings_path)
    
    if embeddings is None:
        return
    
    # Выбираем эмбединги в зависимости от наличия PCA
    if args.use_pca and embeddings_pca is not None:
        logger.info("Используются PCA-эмбединги")
        X = embeddings_pca
    else:
        X = embeddings
    
    # Подготовка меток
    label_column = get_label_column(args.task, metadata)
    
    if metadata is None or label_column not in metadata.columns:
        logger.error(f"Не найдена колонка с метками: {label_column}")
        return
    
    y = metadata[label_column].values
    
    # Если метки текстовые, кодируем их числовыми значениями
    if y.dtype == object:
        logger.info("Преобразование текстовых меток в числовые")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        # Сохраняем маппинг для дальнейшего использования
        label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
        logger.info(f"Маппинг меток: {label_mapping}")
        
        # Сохраняем label_encoder для использования при генерации предсказаний
        os.makedirs(os.path.join("..", "results", "encoders"), exist_ok=True)
        with open(os.path.join("..", "results", "encoders", f"task{args.task}_label_encoder.pkl"), 'wb') as f:
            pickle.dump(label_encoder, f)
        logger.info(f"LabelEncoder сохранен для задачи {args.task}")
    
    # Разделение на тренировочную и валидационную выборки по метке 'split'
    if 'split' in metadata.columns:
        logger.info("Разделение на тренировочную и валидационную выборки по метке 'split'")
        train_mask = metadata['split'] == 'train'
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[~train_mask], y[~train_mask]
    else:
        # Если нет метки 'split', используем стратифицированное разбиение
        logger.info("Разделение на тренировочную и валидационную выборки с помощью StratifiedKFold")
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    logger.info(f"Размер тренировочной выборки: {X_train.shape[0]}")
    logger.info(f"Размер валидационной выборки: {X_val.shape[0]}")
    
    # Получение моделей
    models = get_models(args.task, balance=args.balance)
    
    # Обучение и оценка моделей
    results = train_and_evaluate(
        X_train, y_train, X_val, y_val, 
        models, args.model_type, grid_search=args.grid_search
    )
    
    if results is None:
        return
    
    # Сохранение результатов
    metrics_path = save_results(results, args.output_dir, args.task)
    
    # Построение графиков с результатами
    plot_results(results, args.output_dir, args.task)
    
    logger.info("Обучение и оценка моделей завершены успешно.")

if __name__ == "__main__":
    main()
    