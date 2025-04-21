import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from model import train_model, predict
import joblib
from sklearn.preprocessing import StandardScaler
import os
from config import DATASET_PATH

# Настройка страницы
st.set_page_config(
    page_title="Анализ качества абитуриентов",
    page_icon="🎓",
    layout="wide"
)

# Заголовок приложения
st.title("Анализ качества абитуриентов на основе прогнозирования")
st.markdown("""
    Эта информационная система использует методы прогнозирования для оценки качества абитуриентов.
""")

# Загрузка данных
@st.cache_data
def load_app_data():
    data = pd.read_csv(DATASET_PATH, sep=';', decimal=',')
    return data

# Загрузка модели и трансформаций
@st.cache_resource
def load_model():
    if not os.path.exists('optimized_model.joblib'):
        # Если модель не существует, обучаем новую
        data = pd.read_csv('abiturients_dataset.csv', sep=';', decimal=',')
        model, accuracy, cm, feature_importance = train_model(data)
    else:
        # Загружаем сохраненную модель
        model = joblib.load('optimized_model.joblib')
        scaler = joblib.load('scaler.joblib')
        label_encoders = joblib.load('label_encoders.joblib')
    return model, scaler, label_encoders

# Инициализация данных и моделей
if 'data' not in st.session_state:
    st.session_state['data'] = load_app_data()
    st.session_state['model'], st.session_state['scaler'], st.session_state['label_encoders'] = load_model()

# Сайдбар для навигации
st.sidebar.title("Навигация")
page = st.sidebar.radio(
    "Выберите раздел:",
    ["Обзор данных", "Анализ данных", "Прогнозирование"]
)

# Функция для отображения обзора данных
def show_data_overview():
    st.header("Обзор данных")
    df = st.session_state['data']
    
    st.write("Первые 5 строк датасета:")
    st.dataframe(df.head())
    
    st.write("Описательная статистика:")
    st.write(df.describe())

# Функция для анализа данных
def analyze_data():
    st.header("Анализ данных")
    df = st.session_state['data']
    
    # Выбор параметров для анализа
    analysis_type = st.selectbox(
        "Выберите тип анализа:",
        ["Описательная статистика", "Визуализация", "Корреляционный анализ"]
    )
    
    if analysis_type == "Описательная статистика":
        st.subheader("Описательная статистика")
        st.write(df.describe())
        
    elif analysis_type == "Визуализация":
        st.subheader("Визуализация данных")
        chart_type = st.selectbox(
            "Выберите тип графика:",
            ["Гистограмма", "Диаграмма рассеяния", "Ящик с усами"]
        )
        
        if chart_type == "Гистограмма":
            column = st.selectbox("Выберите колонку для анализа:", df.columns)
            fig = px.histogram(df, x=column, title=f"Распределение {column}",
                             color_discrete_sequence=['#66B2FF'])
            st.plotly_chart(fig)
            
        elif chart_type == "Диаграмма рассеяния":
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox("Выберите переменную для оси X:", 
                                      [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
            with col2:
                y_column = st.selectbox("Выберите переменную для оси Y:", 
                                      [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
            
            color_column = st.selectbox("Выберите категорию для раскраски точек:", 
                                      ['Нет'] + [col for col in df.columns if col not in [x_column, y_column]])
            
            if color_column != 'Нет':
                fig = px.scatter(df, x=x_column, y=y_column, color=color_column,
                               title=f"Диаграмма рассеяния: {x_column} vs {y_column}",
                               color_discrete_sequence=['#66B2FF', '#FF9999', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF'])
            else:
                fig = px.scatter(df, x=x_column, y=y_column,
                               title=f"Диаграмма рассеяния: {x_column} vs {y_column}",
                               color_discrete_sequence=['#66B2FF'])
            
            st.plotly_chart(fig)
            
        elif chart_type == "Ящик с усами":
            column = st.selectbox("Выберите колонку для анализа:", 
                                [col for col in df.columns if df[col].dtype in ['int64', 'float64']])
            group_column = st.selectbox("Выберите колонку для группировки:", 
                                      ['Нет'] + [col for col in df.columns if col != column])
            
            if group_column != 'Нет':
                fig = px.box(df, x=group_column, y=column, 
                           title=f"Ящик с усами: {column} по {group_column}",
                           color_discrete_sequence=['#66B2FF', '#99CCFF'])
            else:
                fig = px.box(df, y=column, title=f"Ящик с усами: {column}",
                           color_discrete_sequence=['#66B2FF'])
            
            st.plotly_chart(fig)
            
    elif analysis_type == "Корреляционный анализ":
        st.subheader("Корреляционный анализ")
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(corr_matrix, title="Матрица корреляций",
                       color_continuous_scale=[[0, '#3B4992'],
                                            [0.5, '#FFFFFF'],
                                            [1, '#B40426']])
        st.plotly_chart(fig)

# Функция для прогнозирования
def show_prediction():
    st.header("Прогнозирование успешности")
    
    st.subheader("Проверьте свои шансы на поступление")
    st.markdown("""
        Введите ваши данные, и система оценит вероятность вашего поступления.
        Все поля обязательны для заполнения.
    """)

    col1, col2 = st.columns(2)
    
    with col1:
        gpa = st.number_input("Средний балл аттестата", min_value=2.0, max_value=5.0, value=4.0, step=0.01)
        total_score = st.number_input("Общий балл ЕГЭ", min_value=100, max_value=300, value=200)
        individual_achievements = st.number_input("Баллы за индивидуальные достижения", min_value=0, max_value=10, value=0)
        
    with col2:
        faculty = st.selectbox("Выберите факультет", [
            "Информатика", "Экономика", "Юриспруденция", "Медицина", 
            "Физика", "Биология", "Математика", "Филология"
        ])
        study_form = st.selectbox("Форма обучения", ["Очная", "Заочная"])
        admission_category = st.selectbox("Категория поступления", 
            ["Общий конкурс", "Особая квота", "Целевая квота"]
        )

    if st.button("Получить прогноз", type="primary"):
        # Подготовка данных для предсказания
        input_data = pd.DataFrame({
            'Gender': ['М'],  # Значение по умолчанию
            'Age': [18],  # Значение по умолчанию
            'Admission_Year': [2024],
            'GPA': [gpa],
            'Total_Score': [total_score],
            'Math_EGE': [total_score // 3],
            'Russian_EGE': [total_score // 3],
            'Profile_EGE': [total_score // 3],
            'Individual_Achievements': [individual_achievements],
            'Admission_Category': [admission_category],
            'Study_Form': [study_form],
            'Faculty': [faculty]
        })

        # Кодирование категориальных переменных
        for column, encoder in st.session_state['label_encoders'].items():
            if column in input_data.columns:
                input_data[column] = encoder.transform(input_data[column])

        # Применяем веса к признакам
        feature_weights = {
            'GPA': 3.0,  # Сильно увеличиваем влияние среднего балла
            'Total_Score': 3.0,  # Сильно увеличиваем влияние общего балла
            'Individual_Achievements': 2.0,  # Увеличиваем влияние индивидуальных достижений
            'Math_EGE': 1.5,  # Увеличиваем влияние профильного предмета
            'Profile_EGE': 1.5  # Увеличиваем влияние профильного предмета
        }
        
        for feature, weight in feature_weights.items():
            if feature in input_data.columns:
                input_data[feature] = input_data[feature] * weight

        # Нормализация данных
        input_data_scaled = st.session_state['scaler'].transform(input_data)
        
        # Получаем предсказание и вероятности
        prediction = st.session_state['model'].predict(input_data_scaled)
        probabilities = st.session_state['model'].predict_proba(input_data_scaled)
        
        # Выводим результат
        st.subheader("Результат прогноза:")
        
        # Анализ факторов
        factors = []
        
        # Анализируем конкурс
        competition = st.session_state['data'][st.session_state['data']['Faculty'] == faculty]['Competition_Per_Seat'].mean()
        if competition > 5:
            factors.append(f"⚠️ Высокий конкурс на выбранном факультете: {competition:.2f} человек на место")
        else:
            factors.append(f"✅ Умеренный конкурс на выбранном факультете: {competition:.2f} человек на место")
        
        # Анализируем форму обучения
        if study_form == "Очная":
            factors.append("⚠️ Очная форма обучения обычно более конкурентная")
        else:
            factors.append("✅ Заочная форма обучения обычно менее конкурентная")
        
        # Анализируем категорию поступления
        if admission_category == "Общий конкурс":
            factors.append("⚠️ Общий конкурс - самая конкурентная категория поступления")
        else:
            factors.append(f"✅ {admission_category} увеличивает шансы на поступление")
        
        # Анализируем баллы
        if total_score >= 270:
            factors.append("✅ Высокий общий балл ЕГЭ")
        elif total_score >= 240:
            factors.append("ℹ️ Средний общий балл ЕГЭ")
        else:
            factors.append("⚠️ Относительно низкий общий балл ЕГЭ")
        
        if individual_achievements >= 8:
            factors.append("✅ Высокий балл за индивидуальные достижения")
        elif individual_achievements >= 5:
            factors.append("ℹ️ Средний балл за индивидуальные достижения")
        else:
            factors.append("⚠️ Низкий балл за индивидуальные достижения")

        # Выводим вероятность поступления
        admission_probability = probabilities[0][1]
        if admission_probability >= 0.7:
            st.success(f"🎉 Высокая вероятность поступления: {admission_probability:.1%}")
        elif admission_probability >= 0.4:
            st.warning(f"⚠️ Средняя вероятность поступления: {admission_probability:.1%}")
        else:
            st.error(f"❌ Низкая вероятность поступления: {admission_probability:.1%}")
        
        st.subheader("Анализ факторов:")
        for factor in factors:
            st.write(factor)
        
        # Рекомендации
        st.subheader("Рекомендации:")
        recommendations = []
        
        if admission_probability < 0.7:  # Если вероятность поступления не очень высокая
            if admission_category == "Общий конкурс":
                recommendations.append("• Рассмотрите возможность поступления по целевой квоте")
            if study_form == "Очная":
                recommendations.append("• Рассмотрите возможность поступления на заочную форму обучения")
            if total_score < 250:
                recommendations.append("• Повысьте общий балл ЕГЭ - это один из ключевых факторов успеха")
            if individual_achievements < 5:
                recommendations.append("• Получите дополнительные баллы за индивидуальные достижения")
            if competition > 5:
                recommendations.append("• Рассмотрите факультеты с меньшим конкурсом")
        
        if recommendations:
            for rec in recommendations:
                st.write(rec)

# Основная логика приложения
if page == "Обзор данных":
    show_data_overview()
elif page == "Анализ данных":
    analyze_data()
elif page == "Прогнозирование":
    show_prediction()
