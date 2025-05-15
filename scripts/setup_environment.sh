#!/bin/bash
# Скрипт для настройки окружения и установки зависимостей

# Проверяем, установлен ли conda
if ! command -v conda &> /dev/null; then
    echo "Conda не установлена. Пожалуйста, установите Miniconda или Anaconda."
    echo "Инструкции по установке: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html"
    exit 1
fi

# Создаем conda окружение из environment.yml
echo "Создание conda окружения 'genai-detection'..."
conda env create -f environment.yml
if [ $? -ne 0 ]; then
    echo "Не удалось создать conda окружение. Попробуем обновить существующее..."
    conda env update -f environment.yml
fi

# Активируем окружение
echo "Активация окружения 'genai-detection'..."
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS или Linux
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate genai-detection
else
    # Windows
    echo "На Windows, пожалуйста, запустите 'conda activate genai-detection' вручную"
fi

# Проверяем установку gdown
echo "Проверка установки gdown..."
if conda run -n genai-detection python -c "import gdown" 2>/dev/null; then
    echo "gdown уже установлен."
else
    echo "Установка gdown..."
    conda run -n genai-detection pip install gdown
fi

# Информация об установленных пакетах
echo "Установленные версии пакетов:"
conda run -n genai-detection pip list | grep -E "pandas|numpy|matplotlib|gdown|torch|transformers"

echo "Настройка окружения завершена."
echo ""
echo "Для активации окружения используйте команду:"
echo "conda activate genai-detection"
echo ""
echo "Для запуска скрипта скачивания данных:"
echo "bash download_all_data.sh"