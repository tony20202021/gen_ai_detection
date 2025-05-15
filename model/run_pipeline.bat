@echo off
REM Скрипт для запуска полного пайплайна обнаружения генеративного ИИ (Windows)

REM Цвета для вывода в Windows (не поддерживаются)
SETLOCAL EnableDelayedExpansion

REM Функция для вывода статусов
:log
    echo [%date% %time%] %~1
    goto :EOF

:error
    echo [%date% %time%] ОШИБКА: %~1
    goto :EOF

:success
    echo [%date% %time%] УСПЕХ: %~1
    goto :EOF

:warning
    echo [%date% %time%] ВНИМАНИЕ: %~1
    goto :EOF

REM Проверка активации conda окружения
IF NOT "%CONDA_DEFAULT_ENV%"=="genai-detection" (
    call :warning "Окружение conda 'genai-detection' не активировано!"
    set /p choice="Хотите активировать окружение? (y/n): "
    if /i "!choice!"=="y" (
        call conda activate genai-detection
        if !ERRORLEVEL! NEQ 0 (
            call :error "Не удалось активировать окружение. Запустите скрипт setup_environment.bat."
            exit /b 1
        )
    ) else (
        call :warning "Продолжение без активации окружения conda. Возможны ошибки!"
    )
)

REM Параметры по умолчанию
set TASK=1
set MODEL=all-mpnet-base-v2
set BATCH_SIZE=8
set MAX_LENGTH=512
set CLUSTERING_METHOD=kmeans
set USE_PCA=true
set BALANCE=true
set MODEL_TYPE=all
set GRID_SEARCH=false

REM Парсинг параметров
:parse_args
if "%~1"=="" goto :args_done
if "%~1"=="-t" (
    set TASK=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--task" (
    set TASK=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-m" (
    set MODEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-b" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-l" (
    set MAX_LENGTH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max-length" (
    set MAX_LENGTH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="-c" (
    set CLUSTERING_METHOD=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--clustering" (
    set CLUSTERING_METHOD=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-pca" (
    set USE_PCA=false
    shift
    goto :parse_args
)
if "%~1"=="--no-balance" (
    set BALANCE=false
    shift
    goto :parse_args
)
if "%~1"=="--model-type" (
    set MODEL_TYPE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--grid-search" (
    set GRID_SEARCH=true
    shift
    goto :parse_args
)
if "%~1"=="-h" (
    goto :show_help
)
if "%~1"=="--help" (
    goto :show_help
)
call :error "Неизвестный параметр: %~1"
goto :show_help

:show_help
echo Использование: run_pipeline.bat [опции]
echo Опции:
echo   -t, --task NUMBER         Номер задачи (1 - бинарная, 2 - многоклассовая, по умолчанию: 1)
echo   -m, --model MODEL         Модель для извлечения эмбедингов (по умолчанию: all-mpnet-base-v2)
echo   -b, --batch-size SIZE     Размер батча для обработки текстов (по умолчанию: 8)
echo   -l, --max-length LENGTH   Максимальная длина текста в токенах (по умолчанию: 512)
echo   -c, --clustering METHOD   Метод кластеризации (kmeans, dbscan, agglomerative, по умолчанию: kmeans)
echo   --no-pca                  Не использовать PCA для понижения размерности
echo   --no-balance              Не учитывать несбалансированность классов
echo   --model-type TYPE         Тип модели для обучения (logistic, svm, rf, xgb, mlp, all, по умолчанию: all)
echo   --grid-search             Выполнить поиск гиперпараметров с помощью GridSearchCV
echo   -h, --help                Показать эту справку
echo.
echo Пример: run_pipeline.bat -t 2 -m sentence-transformers/all-mpnet-base-v2 --grid-search
exit /b 1

:args_done

REM Проверка наличия данных
if not exist "data\task%TASK%" (
    call :warning "Директория с данными для задачи %TASK% не найдена."
    set /p choice="Хотите скачать данные сейчас? (y/n): "
    if /i "!choice!"=="y" (
        call :log "Скачивание данных для задачи %TASK%..."
        python download_task%TASK%_data.py
    ) else (
        call :error "Данные для задачи %TASK% отсутствуют. Запустите скрипт download_task%TASK%_data.py."
        exit /b 1
    )
)

REM Создание необходимых директорий
call :log "Создание необходимых директорий..."
if not exist "data\embeddings" mkdir data\embeddings
if not exist "data\clustering" mkdir data\clustering
if not exist "models" mkdir models
if not exist "results" mkdir results

REM Шаг 1: Извлечение эмбедингов
call :log "Шаг 1: Извлечение эмбедингов из текстов..."
set MODEL_NAME=%MODEL:.=_%
set MODEL_NAME=%MODEL_NAME:/=_%
set MODEL_NAME=%MODEL_NAME:\=_%
set MODEL_NAME=%MODEL_NAME: =_%
set EMBEDDINGS_PATH=data\embeddings\task%TASK%_embeddings_%MODEL_NAME%.pkl
set EMBEDDINGS_WITH_DIM_PATH=data\embeddings\task%TASK%_embeddings_%MODEL_NAME%_with_dim_reduction.pkl

REM Проверка наличия файла эмбедингов
if exist "%EMBEDDINGS_WITH_DIM_PATH%" (
    call :warning "Файл эмбедингов с пониженной размерностью уже существует: %EMBEDDINGS_WITH_DIM_PATH%"
    set /p choice="Пересоздать файл? (y/n): "
    if /i "!choice!"=="y" (
        REM Запуск извлечения эмбедингов
        call :log "Извлечение эмбедингов с моделью: %MODEL%"
        python extract_embeddings.py --task %TASK% --model %MODEL% --batch_size %BATCH_SIZE% --max_length %MAX_LENGTH% --output_dir data\embeddings --reduce_dim
        call :success "Эмбединги успешно извлечены"
    ) else (
        call :success "Используем существующий файл эмбедингов"
    )
) else (
    REM Запуск извлечения эмбедингов
    call :log "Извлечение эмбедингов с моделью: %MODEL%"
    python extract_embeddings.py --task %TASK% --model %MODEL% --batch_size %BATCH_SIZE% --max_length %MAX_LENGTH% --output_dir data\embeddings --reduce_dim
    call :success "Эмбединги успешно извлечены"
)

REM Шаг 2: Кластеризация эмбедингов
call :log "Шаг 2: Кластеризация эмбедингов методом %CLUSTERING_METHOD%..."
set CLUSTERING_PATH=data\clustering\task%TASK%_%CLUSTERING_METHOD%_clustering_results.pkl

REM Проверка наличия файла результатов кластеризации
if exist "%CLUSTERING_PATH%" (
    call :warning "Файл результатов кластеризации уже существует: %CLUSTERING_PATH%"
    set /p choice="Пересоздать файл? (y/n): "
    if /i "!choice!"=="y" (
        REM Запуск кластеризации
        call :log "Выполнение кластеризации методом %CLUSTERING_METHOD%..."
        
        set CMD=python clustering.py --task %TASK% --embeddings_path %EMBEDDINGS_WITH_DIM_PATH% --output_dir data\clustering --method %CLUSTERING_METHOD%
        
        if "%USE_PCA%"=="true" (
            set CMD=%CMD% --use_pca
        )
        
        %CMD%
        call :success "Кластеризация успешно завершена"
    ) else (
        call :success "Используем существующий файл результатов кластеризации"
    )
) else (
    REM Запуск кластеризации
    call :log "Выполнение кластеризации методом %CLUSTERING_METHOD%..."
    
    set CMD=python clustering.py --task %TASK% --embeddings_path %EMBEDDINGS_WITH_DIM_PATH% --output_dir data\clustering --method %CLUSTERING_METHOD%
    
    if "%USE_PCA%"=="true" (
        set CMD=%CMD% --use_pca
    )
    
    %CMD%
    call :success "Кластеризация успешно завершена"
)

REM Шаг 3: Обучение и оценка моделей
call :log "Шаг 3: Обучение и оценка моделей классификации..."

REM Формирование команды с параметрами
set TRAIN_CMD=python train_models.py --task %TASK% --embeddings_path %EMBEDDINGS_WITH_DIM_PATH% --output_dir models --model_type %MODEL_TYPE%

if "%USE_PCA%"=="true" (
    set TRAIN_CMD=%TRAIN_CMD% --use_pca
)

if "%BALANCE%"=="true" (
    set TRAIN_CMD=%TRAIN_CMD% --balance
)

if "%GRID_SEARCH%"=="true" (
    set TRAIN_CMD=%TRAIN_CMD% --grid_search
)

REM Запуск обучения моделей
call :log "Выполнение команды: %TRAIN_CMD%"
%TRAIN_CMD%

if %ERRORLEVEL% EQU 0 (
    call :success "Обучение и оценка моделей успешно завершены"
) else (
    call :error "Ошибка при обучении моделей"
    exit /b 1
)

REM Шаг 4: Визуализация результатов
call :log "Шаг 4: Визуализация эмбедингов и результатов в Jupyter Notebook..."
call :log "Для визуализации результатов запустите Jupyter Notebook с файлом visualize_embeddings.ipynb:"
call :log "    jupyter notebook visualize_embeddings.ipynb"

call :success "Пайплайн успешно выполнен!"
echo.
echo Созданные файлы и директории:
echo   Эмбединги: %EMBEDDINGS_WITH_DIM_PATH%
echo   Результаты кластеризации: %CLUSTERING_PATH%
echo   Модели: models\task%TASK%_*_model_*.joblib
echo   Метрики: models\task%TASK%_metrics_*.json
echo.
echo Следующие шаги:
echo   1. Просмотрите визуализацию эмбедингов и кластеров в Jupyter Notebook
echo   2. Изучите метрики моделей и выберите лучшую для вашей задачи
echo   3. Примените выбранную модель к новым данным для определения генеративного ИИ

exit /b 0
