#!/usr/bin/env python3
"""
Скрипт для кластеризации эмбедингов текстов и анализа результатов
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import argparse
import logging
from tqdm import tqdm
import joblib
from collections import Counter

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clustering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Кластеризация эмбедингов текстов')
    parser.add_argument('--task', type=int, choices=[1, 2], required=True, 
                        help='Номер задачи (1 - бинарная классификация, 2 - многоклассовая)')
    parser.add_argument('--embeddings_path', type=str, required=True,
                        help='Путь к файлу с эмбедингами')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Директория для сохранения результатов (по умолчанию: ../data/clustering)')
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Количество кластеров (по умолчанию: число уникальных меток)')
    parser.add_argument('--method', type=str, default='kmeans',
                        choices=['kmeans', 'dbscan', 'agglomerative'],
                        help='Метод кластеризации')
    parser.add_argument('--use_pca', action='store_true',
                        help='Использовать PCA-эмбединги, если доступны')
    args = parser.parse_args()
    
    # Установка значения по умолчанию для output_dir, если не указано
    if args.output_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        args.output_dir = os.path.join(base_dir, "data", "clustering")
    
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
            embeddings_tsne = data.get("embeddings_tsne")
        else:
            embeddings = data
            metadata = None
            embeddings_pca = None
            embeddings_tsne = None
            
        logger.info(f"Загружено {embeddings.shape[0]} эмбедингов размерности {embeddings.shape[1]}")
        return embeddings, metadata, embeddings_pca, embeddings_tsne
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке эмбедингов: {e}")
        return None, None, None, None

def get_label_column(task, metadata):
    """Определение колонки с метками в зависимости от задачи"""
    if task == 1:
        label_column = 'label'
    else:  # task == 2
        label_column = 'label_text' if 'label_text' in metadata.columns else 'label'
    
    return label_column

def perform_clustering(embeddings, n_clusters, method='kmeans'):
    """Выполнение кластеризации с использованием выбранного метода"""
    logger.info(f"Применение {method} для кластеризации")
    
    # Стандартизация данных
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = clusterer.fit_predict(embeddings_scaled)
        
    elif method == 'dbscan':
        # Подбор параметра eps на основе размерности данных
        eps = 0.5 * np.sqrt(embeddings.shape[1] / 100)
        clusterer = DBSCAN(eps=eps, min_samples=5)
        cluster_labels = clusterer.fit_predict(embeddings_scaled)
        
        # Обработка выбросов (метка -1)
        if -1 in cluster_labels:
            outliers_count = np.sum(cluster_labels == -1)
            logger.info(f"DBSCAN выявил {outliers_count} выбросов ({outliers_count/len(cluster_labels)*100:.2f}%)")
        
    elif method == 'agglomerative':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(embeddings_scaled)
    
    logger.info(f"Кластеризация завершена. Распределение кластеров: {Counter(cluster_labels)}")
    return cluster_labels, clusterer, scaler

def evaluate_clustering(embeddings, cluster_labels, true_labels=None):
    """Оценка качества кластеризации"""
    evaluation_results = {}
    
    # Силуэтный коэффициент
    try:
        if len(np.unique(cluster_labels)) > 1:
            silhouette = silhouette_score(embeddings, cluster_labels)
            evaluation_results['silhouette_score'] = silhouette
            logger.info(f"Силуэтный коэффициент: {silhouette:.4f}")
        else:
            logger.warning("Невозможно вычислить силуэтный коэффициент: все объекты отнесены к одному кластеру")
    except Exception as e:
        logger.error(f"Ошибка при вычислении силуэтного коэффициента: {e}")
    
    # Оценка согласованности с истинными метками (если доступны)
    if true_labels is not None:
        try:
            ari = adjusted_rand_score(true_labels, cluster_labels)
            evaluation_results['adjusted_rand_index'] = ari
            logger.info(f"Adjusted Rand Index: {ari:.4f}")
            
            # Вычисление матрицы несоответствия
            cm = confusion_matrix(true_labels, cluster_labels)
            evaluation_results['confusion_matrix'] = cm
        except Exception as e:
            logger.error(f"Ошибка при сравнении с истинными метками: {e}")
    
    return evaluation_results

def save_results(results, output_dir, task, method):
    """Сохранение результатов кластеризации"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"task{task}_{method}_clustering_results.pkl")
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Результаты кластеризации сохранены в {output_path}")

def main():
    args = parse_args()
    
    # Создаем директорию для сохранения результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Загружаем эмбединги
    embeddings, metadata, embeddings_pca, embeddings_tsne = load_embeddings(args.embeddings_path)
    
    if embeddings is None:
        return
    
    # Выбираем эмбединги в зависимости от наличия PCA
    if args.use_pca and embeddings_pca is not None:
        logger.info("Используются PCA-эмбединги")
        embeddings_for_clustering = embeddings_pca
    else:
        embeddings_for_clustering = embeddings
    
    # Определяем количество кластеров (если не указано)
    if args.n_clusters is None:
        if metadata is not None:
            label_column = get_label_column(args.task, metadata)
            unique_labels = metadata[label_column].nunique()
            n_clusters = unique_labels
            logger.info(f"Количество кластеров установлено по количеству уникальных меток: {n_clusters}")
        else:
            n_clusters = 2 if args.task == 1 else 6
            logger.info(f"Установлено количество кластеров по умолчанию для задачи {args.task}: {n_clusters}")
    else:
        n_clusters = args.n_clusters
    
    # Выполняем кластеризацию
    cluster_labels, clusterer, scaler = perform_clustering(
        embeddings_for_clustering, n_clusters, method=args.method
    )
    
    # Оцениваем качество кластеризации
    true_labels = None
    if metadata is not None:
        label_column = get_label_column(args.task, metadata)
        if label_column in metadata.columns:
            if args.task == 1:
                # Для бинарной задачи метки обычно числовые
                true_labels = metadata[label_column].values
            else:
                # Для многоклассовой задачи преобразуем текстовые метки в числовые
                label_mapping = {label: i for i, label in enumerate(metadata[label_column].unique())}
                true_labels = metadata[label_column].map(label_mapping).values
    
    evaluation_results = evaluate_clustering(embeddings_for_clustering, cluster_labels, true_labels)
    
    # Сохраняем результаты
    results = {
        "cluster_labels": cluster_labels,
        "clusterer": clusterer,
        "scaler": scaler,
        "evaluation": evaluation_results,
        "metadata": metadata,
        "embeddings_tsne": embeddings_tsne,
        "method": args.method,
        "task": args.task,
        "n_clusters": n_clusters
    }
    
    save_results(results, args.output_dir, args.task, args.method)

if __name__ == "__main__":
    main()
    