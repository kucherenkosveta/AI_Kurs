import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
from config import DATASET_PATH
import joblib
from sklearn.linear_model import LogisticRegression

# Загрузка данных и предобработка (если нужно)
def load_data():
    data = pd.read_csv(DATASET_PATH, sep=';', decimal=',')
    return data

# Функция для кодирования категориальных переменных
def encode_categorical(data):
    # Создаем копию датасета
    df = data.copy()
    
    # Список категориальных колонок
    categorical_columns = ['Gender', 'Admission_Category', 'Study_Form', 'Faculty']
    
    # Создаем и применяем LabelEncoder для каждой категориальной колонки
    label_encoders = {}
    for column in categorical_columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    
    return df, label_encoders

# Функция для анализа корреляций
def analyze_correlations(df):
    # Вычисляем корреляции
    correlations = df.corr()
    
    # Создаем тепловую карту корреляций
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
    plt.title('Матрица корреляций')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    return correlations

# Функция обучения модели
def train_model(data=None):
    if data is None:
        data = load_data()
    
    # Кодируем категориальные переменные
    df, label_encoders = encode_categorical(data)
    
    # Разделяем на признаки и целевую переменную
    X = df.drop(['Enrolled', 'Competition_Per_Seat'], axis=1)  # Убираем конкуренцию
    y = df['Enrolled']
    
    # Создаем веса для признаков
    feature_weights = {
        'GPA': 3.0,  # Сильно увеличиваем влияние среднего балла
        'Total_Score': 3.0,  # Сильно увеличиваем влияние общего балла
        'Individual_Achievements': 2.0,  # Увеличиваем влияние индивидуальных достижений
        'Math_EGE': 1.5,  # Увеличиваем влияние профильного предмета
        'Profile_EGE': 1.5  # Увеличиваем влияние профильного предмета
    }
    
    # Применяем веса к признакам
    for feature, weight in feature_weights.items():
        if feature in X.columns:
            X[feature] = X[feature] * weight
    
    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Создаем модель с весами классов
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    
    # Обучаем модель
    model.fit(X_train_scaled, y_train)
    
    # Получаем предсказания
    y_pred = model.predict(X_test_scaled)
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Выводим отчет о классификации
    print("\nОтчет о классификации:")
    print(classification_report(y_test, y_pred))
    
    # Анализируем важность признаков
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    print("\nВажность признаков:")
    print(feature_importance)
    
    # Сохраняем модель и необходимые трансформации
    joblib.dump(model, 'optimized_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    
    return model, accuracy, cm, feature_importance

# Функция для предсказания
def predict(model, data):
    return model.predict(data)

# Дополнительные функции для анализа
def get_confusion_matrix(cm):
    TN, FP, FN, TP = cm.ravel()
    precision_class_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    recall_class_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    precision_class_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_class_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    return precision_class_0, recall_class_0, precision_class_1, recall_class_1

if __name__ == '__main__':
    # Загружаем данные
    data = load_data()
    
    # Обучаем модель
    model, accuracy, cm, feature_importance = train_model(data)
    
    print(f"\nТочность модели: {accuracy:.2f}")
    
    # Выводим метрики для классов
    precision_class_0, recall_class_0, precision_class_1, recall_class_1 = get_confusion_matrix(cm)
    print("\nМетрики для классов:")
    print(f"Точность класса 0: {precision_class_0:.2f}")
    print(f"Полнота класса 0: {recall_class_0:.2f}")
    print(f"Точность класса 1: {precision_class_1:.2f}")
    print(f"Полнота класса 1: {recall_class_1:.2f}")
    
    # Проверяем на абитуриенте с идеальными баллами
    df, label_encoders = encode_categorical(data)
    scaler = StandardScaler()
    X = df.drop(['Enrolled', 'Competition_Per_Seat'], axis=1)  # Убираем конкуренцию
    scaler.fit(X)
    
    # Создаем тестовый пример с идеальными баллами
    test_data = pd.DataFrame({
        'Gender': [0],  # М
        'Age': [18],
        'Admission_Year': [2024],
        'GPA': [5.0],
        'Total_Score': [300],
        'Math_EGE': [100],
        'Russian_EGE': [100],
        'Profile_EGE': [100],
        'Individual_Achievements': [10],
        'Admission_Category': [0],  # Общий конкурс
        'Study_Form': [1],  # Очная
        'Faculty': [0]  # Информатика
    })
    
    # Применяем веса к признакам
    feature_weights = {
        'GPA': 3.0,
        'Total_Score': 3.0,
        'Individual_Achievements': 2.0,
        'Math_EGE': 1.5,
        'Profile_EGE': 1.5
    }
    
    for feature, weight in feature_weights.items():
        if feature in test_data.columns:
            test_data[feature] = test_data[feature] * weight
    
    # Нормализуем данные
    test_data_scaled = scaler.transform(test_data)
    
    # Получаем предсказание и вероятности
    prediction = model.predict(test_data_scaled)
    probabilities = model.predict_proba(test_data_scaled)
    
    print("\nТест на абитуриенте с идеальными баллами:")
    print(f"Предсказание: {'Поступил' if prediction[0] == 1 else 'Не поступил'}")
    print(f"Вероятность поступления: {probabilities[0][1]:.2%}")
