import streamlit as st
import pandas as pd
import numpy as np
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
import networkx as nx
import time
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import geopandas as gpd
from geopy.geocoders import Nominatim

# Функция для загрузки данных


def load_data():
    df = pd.read_csv('cleaned_movies_data.csv')
    return df

# Функция для отображения графиков и анализа


def plot_analysis(df):
    st.title("Анализ данных о фильмах")

    # Гипотеза 1: Бюджет фильмов растет с годами
    st.subheader("Гипотеза 1: Бюджет фильмов растет с годами")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x='release_year',
                 y='budget', estimator='median', ax=ax)
    ax.set_title('Median Movie Budget Over the Years')
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Median Budget')
    st.pyplot(fig)

    # Гипотеза 2: Более высокий бюджет фильмов связан с более высокой выручкой
    st.subheader(
        "Гипотеза 2: Более высокий бюджет фильмов связан с более высокой выручкой")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='budget', y='revenue',
                    hue='release_year', palette='coolwarm', ax=ax)
    sns.regplot(data=df, x='budget', y='revenue',
                scatter=False, color='red', ax=ax)
    ax.set_title('Movie Budget vs. Revenue')
    ax.set_xlabel('Budget')
    ax.set_ylabel('Revenue')
    st.pyplot(fig)

    # Гипотеза 3: Некоторые жанры коррелируют с более высокой выручкой
    st.subheader(
        "Гипотеза 3: Некоторые жанры коррелируют с более высокой выручкой")
    df['genres_list'] = df['genres'].apply(lambda x: x.split(', '))
    genres_revenue = df.explode('genres_list').groupby('genres_list')[
        'revenue'].median().sort_values(ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x=genres_revenue.index, y=genres_revenue.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title('Median Revenue by Genre')
    ax.set_xlabel('Genre')
    ax.set_ylabel('Median Revenue')
    st.pyplot(fig)

    # Корреляционный heatmap
    st.subheader("Корреляционный анализ")
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    fig, ax = plt.subplots()
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)

# Функция для отображения карты


def plot_map(df):
    st.subheader("Географическое распределение фильмов")
    map = folium.Map(location=[20, 0], zoom_start=2)
    marker_cluster = MarkerCluster().add_to(map)

    for index, row in df.iterrows():
        if pd.notnull(row['release_date']):
            release_year = pd.to_datetime(row['release_date']).year
            # Замените на реальные данные
            location = [row['latitude'], row['longitude']]
            folium.Marker(location=location, popup=f"{row['title']} ({release_year})").add_to(
                marker_cluster)

    st._legacy_folium_static(map)

# Функция для машинного обучения


def machine_learning(df):
    st.subheader("Машинное обучение")
    features = ['budget', 'popularity', 'runtime', 'release_year']
    X = df[features]
    y = df['revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = MAE(y_test, y_pred)
    st.write(f'Mean Absolute Error: {mae}')

    # Обучение модели градиентного бустинга
    gbr = GradientBoostingRegressor()
    gbr.fit(X_train, y_train)
    y_pred_gbr = gbr.predict(X_test)
    mae_gbr = MAE(y_test, y_pred_gbr)
    st.write(f'Gradient Boosting MAE: {mae_gbr}')

    # Обучение модели CatBoost
    catboost = CatBoostRegressor(silent=True)
    catboost.fit(X_train, y_train)
    y_pred_catboost = catboost.predict(X_test)
    mae_catboost = MAE(y_test, y_pred_catboost)
    st.write(f'CatBoost MAE: {mae_catboost}')

    # Обучение модели LightGBM
    lgbm = lgb.LGBMRegressor(verbose=-1)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    mae_lgbm = MAE(y_test, y_pred_lgbm)
    st.write(f'LightGBM MAE: {mae_lgbm}')


def main():
    df = load_data()
    st.sidebar.title("Навигация")
    page = st.sidebar.radio(
        "Перейти на", ["Анализ данных", "Карта", "Машинное обучение"])

    if page == "Анализ данных":
        plot_analysis(df)
    elif page == "Карта":
        plot_map(df)
    elif page == "Машинное обучение":
        machine_learning(df)


if __name__ == "__main__":
    main()
