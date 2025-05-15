@echo off
REM Скрипт настройки окружения для Windows

REM Проверяем, установлен ли conda
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Conda не установлена. Пожалуйста, установите Miniconda или Anaconda.
    echo Инструкции по установке: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
    exit /b 1
)

REM Создаем conda окружение из environment.yml
echo Создание conda окружения 'genai-detection'...
conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo Не удалось создать conda окружение. Попробуем обновить существующее...
    conda env update -f environment.yml
)

REM Проверяем установку gdown
echo Проверка установки gdown...
conda run -n genai-detection python -c "import gdown" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Установка gdown...
    conda run -n genai-detection pip install gdown
) else (
    echo gdown уже установлен.
)

REM Информация об установленных пакетах
echo Установленные версии пакетов:
conda run -n genai-detection pip list | findstr /I /C:"pandas" /C:"numpy" /C:"matplotlib" /C:"gdown" /C:"torch" /C:"transformers"

echo Настройка окружения завершена.
echo.
echo Для активации окружения используйте команду:
echo conda activate genai-detection
echo.
echo Для запуска скрипта скачивания данных:
echo python download_task1_data.py
echo python download_task2_data.py