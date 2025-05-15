#!/bin/bash
# Скрипт для запуска полного пайплайна обнаружения генеративного ИИ

set -e  # Остановка скрипта при ошибке

# Цвета для вывода
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Функция для вывода статусов
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ОШИБКА: $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} УСПЕХ: $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} ВНИМАНИЕ: $1"
}

# Проверка активации conda окружения
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "amikhalev_gen_ai_detection" ]; then
    warning "Окружение conda 'amikhalev_gen_ai_detection' не активировано!"
    read -p "Хотите активировать окружение? (y/n): " choice
    if [ "$choice" = "y" ]; then
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate amikhalev_gen_ai_detection
        if [ $? -ne 0 ]; then
            error "Не удалось активировать окружение. Запустите скрипт setup_environment.sh."
            exit 1
        fi
    else
        warning "Продолжение без активации окружения conda. Возможны ошибки!"
    fi
fi

# Параметры по умолчанию
TASK=1  # Номер задачи (1 - бинарная, 2 - многоклассовая)
MODEL="all-mpnet-base-v2"  # Модель для извлечения эмбедингов
BATCH_SIZE=8
MAX_LENGTH=512
CLUSTERING_METHOD="kmeans"
USE_PCA=true
BALANCE=true
MODEL_TYPE="all"
GRID_SEARCH=false

# Вывод помощи
show_help() {
    echo "Использование: ./run_pipeline.sh [опции]"
    echo "Опции:"
    echo "  -t, --task NUMBER         Номер задачи (1 - бинарная, 2 - многоклассовая, по умолчанию: 1)"
    echo "  -m, --model MODEL         Модель для извлечения эмбедингов (по умолчанию: all-mpnet-base-v2)"
    echo "  -b, --batch-size SIZE     Размер батча для обработки текстов (по умолчанию: 8)"
    echo "  -l, --max-length LENGTH   Максимальная длина текста в токенах (по умолчанию: 512)"
    echo "  -c, --clustering METHOD   Метод кластеризации (kmeans, dbscan, agglomerative, по умолчанию: kmeans)"
    echo "  --no-pca                  Не использовать PCA для понижения размерности"
    echo "  --no-balance              Не учитывать несбалансированность классов"
    echo "  --model-type TYPE         Тип модели для обучения (logistic, svm, rf, xgb, mlp, all, по умолчанию: all)"
    echo "  --grid-search             Выполнить поиск гиперпараметров с помощью GridSearchCV"
    echo "  -h, --help                Показать эту справку"
    echo ""
    echo "Пример: ./run_pipeline.sh -t 2 -m sentence-transformers/all-mpnet-base-v2 --grid-search"
}

# Парсинг параметров
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--task)
            TASK="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -l|--max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        -c|--clustering)
            CLUSTERING_METHOD="$2"
            shift 2
            ;;
        --no-pca)
            USE_PCA=false
            shift
            ;;
        --no-balance)
            BALANCE=false
            shift
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --grid-search)
            GRID_SEARCH=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "Неизвестный параметр: $1"
            show_help
            exit 1
            ;;
    esac
done

# Определение путей
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"
DATA_DIR="${BASE_DIR}/data"
MODELS_DIR="${BASE_DIR}/models"
RESULTS_DIR="${BASE_DIR}/results"

# Проверка наличия данных
if [ ! -d "${DATA_DIR}/task${TASK}" ]; then
    warning "Директория с данными для задачи ${TASK} не найдена."
    read -p "Хотите скачать данные сейчас? (y/n): " choice
    if [ "$choice" = "y" ]; then
        log "Скачивание данных для задачи ${TASK}..."
        python "${SCRIPTS_DIR}/download_task${TASK}_data.py"
    else
        error "Данные для задачи ${TASK} отсутствуют. Запустите скрипт ${SCRIPTS_DIR}/download_task${TASK}_data.py."
        exit 1
    fi
fi

# Создание необходимых директорий
log "Создание необходимых директорий..."
mkdir -p "${DATA_DIR}/embeddings"
mkdir -p "${DATA_DIR}/clustering"
mkdir -p "${MODELS_DIR}"
mkdir -p "${RESULTS_DIR}"

# Шаг 1: Извлечение эмбедингов
log "Шаг 1: Извлечение эмбедингов из текстов..."
EMBEDDINGS_PATH="${DATA_DIR}/embeddings/task${TASK}_embeddings_$(basename ${MODEL// /_}).pkl"
EMBEDDINGS_WITH_DIM_PATH="${DATA_DIR}/embeddings/task${TASK}_embeddings_$(basename ${MODEL// /_})_with_dim_reduction.pkl"

# Проверка наличия файла эмбедингов
if [ -f "$EMBEDDINGS_WITH_DIM_PATH" ]; then
    warning "Файл эмбедингов с пониженной размерностью уже существует: $EMBEDDINGS_WITH_DIM_PATH"
    read -p "Пересоздать файл? (y/n): " choice
    if [ "$choice" != "y" ]; then
        success "Используем существующий файл эмбедингов"
    else
        # Запуск извлечения эмбедингов
        log "Извлечение эмбедингов с моделью: $MODEL"
        python extract_embeddings.py --task $TASK --model $MODEL --batch_size $BATCH_SIZE --max_length $MAX_LENGTH --output_dir "${DATA_DIR}/embeddings" --reduce_dim
        success "Эмбединги успешно извлечены"
    fi
else
    # Запуск извлечения эмбедингов
    log "Извлечение эмбедингов с моделью: $MODEL"
    python extract_embeddings.py --task $TASK --model $MODEL --batch_size $BATCH_SIZE --max_length $MAX_LENGTH --output_dir "${DATA_DIR}/embeddings" --reduce_dim
    success "Эмбединги успешно извлечены"
fi

# Шаг 2: Кластеризация эмбедингов
log "Шаг 2: Кластеризация эмбедингов методом $CLUSTERING_METHOD..."
CLUSTERING_PATH="${DATA_DIR}/clustering/task${TASK}_${CLUSTERING_METHOD}_clustering_results.pkl"

# Проверка наличия файла результатов кластеризации
if [ -f "$CLUSTERING_PATH" ]; then
    warning "Файл результатов кластеризации уже существует: $CLUSTERING_PATH"
    read -p "Пересоздать файл? (y/n): " choice
    if [ "$choice" != "y" ]; then
        success "Используем существующий файл результатов кластеризации"
    else
        # Запуск кластеризации
        log "Выполнение кластеризации методом $CLUSTERING_METHOD..."
        if [ "$USE_PCA" = true ]; then
            python clustering.py --task $TASK --embeddings_path $EMBEDDINGS_WITH_DIM_PATH --output_dir "${DATA_DIR}/clustering" --method $CLUSTERING_METHOD --use_pca
        else
            python clustering.py --task $TASK --embeddings_path $EMBEDDINGS_WITH_DIM_PATH --output_dir "${DATA_DIR}/clustering" --method $CLUSTERING_METHOD
        fi
        success "Кластеризация успешно завершена"
    fi
else
    # Запуск кластеризации
    log "Выполнение кластеризации методом $CLUSTERING_METHOD..."
    if [ "$USE_PCA" = true ]; then
        python clustering.py --task $TASK --embeddings_path $EMBEDDINGS_WITH_DIM_PATH --output_dir "${DATA_DIR}/clustering" --method $CLUSTERING_METHOD --use_pca
    else
        python clustering.py --task $TASK --embeddings_path $EMBEDDINGS_WITH_DIM_PATH --output_dir "${DATA_DIR}/clustering" --method $CLUSTERING_METHOD
    fi
    success "Кластеризация успешно завершена"
fi

# Шаг 3: Обучение и оценка моделей
log "Шаг 3: Обучение и оценка моделей классификации..."

# Формирование команды с параметрами
TRAIN_CMD="python train_models.py --task $TASK --embeddings_path $EMBEDDINGS_WITH_DIM_PATH --output_dir ${MODELS_DIR} --model_type $MODEL_TYPE"

if [ "$USE_PCA" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --use_pca"
fi

if [ "$BALANCE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --balance"
fi

if [ "$GRID_SEARCH" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --grid_search"
fi

# Запуск обучения моделей
log "Выполнение команды: $TRAIN_CMD"
$TRAIN_CMD

if [ $? -eq 0 ]; then
    success "Обучение и оценка моделей успешно завершены"
else
    error "Ошибка при обучении моделей"
    exit 1
fi

# Шаг 4: Визуализация результатов
log "Шаг 4: Визуализация эмбедингов и результатов в Jupyter Notebook..."
log "Для визуализации результатов запустите Jupyter Notebook с файлом:"
log "    jupyter notebook ${BASE_DIR}/model/visualize_embeddings.ipynb"

success "Пайплайн успешно выполнен!"
echo ""
echo "Созданные файлы и директории:"
echo "  Эмбединги: $EMBEDDINGS_WITH_DIM_PATH"
echo "  Результаты кластеризации: $CLUSTERING_PATH"
echo "  Модели: ${MODELS_DIR}/task${TASK}_*_model_*.joblib"
echo "  Метрики: ${MODELS_DIR}/task${TASK}_metrics_*.json"
echo ""
echo "Следующие шаги:"
echo "  1. Просмотрите визуализацию эмбедингов и кластеров в Jupyter Notebook"
echo "  2. Изучите метрики моделей и выберите лучшую для вашей задачи"
echo "  3. Примените выбранную модель к новым данным для определения генеративного ИИ"
