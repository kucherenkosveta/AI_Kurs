import os

# Путь к директории с данными
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# Создаем директорию, если она не существует
os.makedirs(DATA_DIR, exist_ok=True)

# Путь к файлу датасета (в текущей директории)
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'abiturients_dataset.csv') 