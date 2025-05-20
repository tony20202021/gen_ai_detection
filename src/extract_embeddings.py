#!/usr/bin/env python3
"""
Скрипт для извлечения эмбедингов из текстов с использованием предобученных моделей
"""

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import argparse
import pickle
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

# Создаем директорию для логов, если её нет
os.makedirs("../logs", exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/embeddings_extraction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Извлечение эмбедингов из текстов')
    parser.add_argument('--task', type=int, choices=[1, 2], required=True, 
                        help='Номер задачи (1 - бинарная классификация, 2 - многоклассовая)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-mpnet-base-v2',
                        help='Название модели для извлечения эмбедингов')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Размер батча для обработки текстов')
    parser.add_argument('--max_length', type=int, default=512, 
                        help='Максимальная длина текста (в токенах)')
    parser.add_argument('--output_dir', type=str, default='../results/embeddings',
                        help='Директория для сохранения эмбедингов')
    parser.add_argument('--reduce_dim', action='store_true',
                        help='Применить понижение размерности (PCA и t-SNE)')
    return parser.parse_args()

def get_data_path(task):
    """Получение путей к файлам данных в зависимости от задачи"""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data")
    
    if task == 1:
        train_path = os.path.join(data_dir, 'task1', 'train.jsonl')
        val_path = os.path.join(data_dir, 'task1', 'val.jsonl')
    else:  # task == 2
        train_path = os.path.join(data_dir, 'task2', 'subtask2_train.jsonl')
        val_path = os.path.join(data_dir, 'task2', 'subtask2_dev.jsonl')
    return train_path, val_path

def load_model_and_tokenizer(model_name):
    """Загрузка предобученной модели и токенизатора"""
    logger.info(f"Загрузка модели и токенизатора: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    model = model.to(device)
    return model, tokenizer, device

def extract_embeddings(texts, model, tokenizer, device, batch_size=8, max_length=512):
    """Извлечение эмбедингов из текстов с использованием предобученной модели"""
    model.eval()
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Извлечение эмбедингов"):
        batch = texts[i:i+batch_size]
        
        # Токенизация с обрезанием и паддингом
        encoded = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        
        # Перемещение тензоров на устройство
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Извлечение эмбедингов без расчета градиентов
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            
        # Используем mean-pooling для получения эмбединга всего текста
        # Суммируем эмбединги последнего слоя и делим на длину последовательности без паддинга
        token_embeddings = outputs.last_hidden_state
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        
        embeddings.append(batch_embeddings)
    
    # Объединение всех батчей
    embeddings = np.vstack(embeddings)
    return embeddings

def reduce_dimensions(embeddings, n_components_pca=50, perplexity=30, n_components_tsne=2):
    """Понижение размерности эмбедингов с использованием PCA и t-SNE"""
    logger.info(f"Применение PCA для снижения размерности до {n_components_pca}")
    pca = PCA(n_components=n_components_pca)
    embeddings_pca = pca.fit_transform(embeddings)
    
    logger.info(f"Применение t-SNE для визуализации (размерность: {n_components_tsne})")
    tsne = TSNE(n_components=n_components_tsne, perplexity=perplexity, n_jobs=-1)
    embeddings_tsne = tsne.fit_transform(embeddings_pca)
    
    return embeddings_pca, embeddings_tsne, pca, tsne

def main():
    args = parse_args()
    
    # Создаем директорию для сохранения эмбедингов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Получаем пути к файлам данных
    train_path, val_path = get_data_path(args.task)
    
    # Загружаем данные
    logger.info(f"Загрузка данных для задачи {args.task}")
    try:
        logger.info(f"train_path {train_path}")
        train_df = pd.read_json(train_path, lines=True)
        logger.info(f"val_path {val_path}")
        val_df = pd.read_json(val_path, lines=True)
        logger.info(f"Загружено {len(train_df)} тренировочных и {len(val_df)} валидационных примеров")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return
    
    # Объединяем данные и добавляем метку источника (train/val)
    train_df["split"] = "train"
    val_df["split"] = "val"
    combined_df = pd.concat([train_df, val_df])
    
    # Загружаем модель и токенизатор
    model, tokenizer, device = load_model_and_tokenizer(args.model)
    
    # Извлекаем эмбединги
    texts = combined_df["text"].tolist()
    embeddings = extract_embeddings(
        texts, model, tokenizer, device, 
        batch_size=args.batch_size, 
        max_length=args.max_length
    )
    
    # Сохраняем оригинальные эмбединги
    output_filename = f"task{args.task}_embeddings_{args.model.split('/')[-1]}"
    embeddings_path = os.path.join(args.output_dir, f"{output_filename}.pkl")
    
    # Сохраняем эмбединги и метаданные
    results = {
        "embeddings": embeddings,
        "metadata": combined_df.drop(columns=["text"]),
        "model_name": args.model,
        "extraction_params": vars(args)
    }
    
    with open(embeddings_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Эмбединги сохранены в {embeddings_path}")
    
    # Понижение размерности, если требуется
    if args.reduce_dim:
        embeddings_pca, embeddings_tsne, pca_model, tsne_model = reduce_dimensions(embeddings)
        
        # Добавляем результаты понижения размерности
        results["embeddings_pca"] = embeddings_pca
        results["embeddings_tsne"] = embeddings_tsne
        
        # Сохраняем модели понижения размерности
        joblib.dump(pca_model, os.path.join(args.output_dir, f"{output_filename}_pca.joblib"))
        
        # Сохраняем обновленные результаты
        with open(os.path.join(args.output_dir, f"{output_filename}_with_dim_reduction.pkl"), 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Результаты понижения размерности сохранены")

if __name__ == "__main__":
    main()
