# Финальный проект: Анализ данных о фильмах

### Дмитрий Головинов
### Студент второго курса
### Совместная программа по экономике НИУ ВШЭ и РЭШ
### Курс: "Наука о данных"

---

## Описание проекта

Этот проект представляет собой аналитическое исследование данных о фильмах. Цель проекта — пройти полный цикл исследования:
- Получить и подгрузить данные
- Обработать пропуски и выбросы
- Сформулировать гипотезы по данным
- Проверить гипотезы статистически/визуально
- Подготовить результаты с выводами о вашей исследовательской работе

---

## Структура проекта

- `final_project.ipynb`: Jupyter ноутбук, содержащий весь рабочий процесс исследования.
- `movies_data.csv`: Сырые данные о фильмах, полученные из API.
- `cleaned_movies_data.csv`: Очищенные и обработанные данные о фильмах.
- `README.md`: Документация проекта (этот файл).

---

## Импортируемые библиотеки и их использование

- **pandas** — Библиотека для работы с табличными данными. Использована для загрузки, обработки и анализа данных.
- **numpy** — Библиотека для численных вычислений. Применена для различных численных операций, таких как вычисление среднего значения.
- **requests** — Библиотека для отправки HTTP-запросов. Использована для взаимодействия с API TMDB и получения данных о фильмах.
- **seaborn** — Библиотека для визуализации данных. Использована для создания различных графиков, таких как линейные графики и диаграммы рассеяния.
- **matplotlib.pyplot** — Библиотека для создания статических и интерактивных визуализаций. Применена для настройки и отображения графиков.
- **folium** — Библиотека для создания интерактивных карт. Использована для визуализации географических данных.
- **networkx** — Библиотека для работы с графами и сетями. Использована для создания и визуализации сети кино сотрудничеств.
- **time** — Встроенная библиотека Python для работы со временем. Применена для управления временем задержки при обращении к API.
- **BeautifulSoup** — Библиотека для парсинга HTML и XML документов. Использована для извлечения данных с веб-страниц IMDb.
- **selenium** — Инструмент для автоматизации веб-браузеров. Применен для более сложного веб-скрейпинга.
- **sklearn** — Набор инструментов для машинного обучения. Использован для подготовки данных, тренировки моделей и оценки их точности.
- **lightgbm** — Библиотека для градиентного бустинга. Применена для построения и тренировки модели машинного обучения.
- **catboost** — Библиотека для градиентного бустинга, оптимизированная для категориальных данных. Использована для построения и тренировки модели машинного обучения.
- **statsmodels** — Библиотека для статистического моделирования. Применена для выполнения статистического анализа.
- **duckdb** — СУБД, встроенная в Python. Использована для выполнения SQL-запросов к DataFrame.
- **warnings** — Встроенная библиотека Python для управления предупреждениями. Применена для подавления предупреждений во время выполнения кода.
- **streamlit** — Фреймворк для создания интерактивных веб-приложений. Использован для создания дашборда для визуализации данных.
- **textblob** — Библиотека для обработки текстов и анализа настроений. Применена для анализа отзывов на фильмы.
- **pandas.json_normalize** — Функция для нормализации JSON данных. Использована для преобразования вложенных структур JSON в табличный формат.

---

## Чекпоинты

### Общие критерии

- [x] **Объем**: Проект содержит 120 строк и больше осмысленного самостоятельно написанного логического кода.
- [x] **Целостность**: Проект выглядит целостным, все технологии используются по делу.
- [x] **Документация**: Качественная документация (которую вы читаете), все компоненты проекта описаны.
- [x] **Качество кода**: Стиль старался по PEP8.
- [x] **Впечатление от проекта**: Вау, круто! (Надеюсь)

### Получение данных

- [x] **Работа с REST API (XML/JSON)**: Использовались API примерно в объеме задач домашней работы.
- [x] **Веб-скреппинг**: Использовался базовый веб-скреппинг с помощью BeautifulSoup.

### Обработка данных

- [x] **Pandas**: Базовое применение.
- [] **Регулярные выражения**: Не использовались.

### Математические возможности Python

- [x] **Использование numpy / scipy**: Содержательное использование numpy для решения математических задач.

### Работа с геоданными

- [x] **Использование geopandas, shapely, folium и т.д.**: Использовались folium и geopy.

### Работа с графами

- [x] **Использование библиотеки networkx**: Использовалась библиотека networkx.

### Визуализация данных

- [x] **Базовые визуализации**: Построен scatter plot, line plot, bar plot.
- [x] **Более сложные визуализации**: Построены сложные визуализации с использованием дополнительных библиотек.

### SQL

- [x] **Использование SQL**: Использовался duckdb для выполнения SQL-запросов к DataFrame.

### Демонстрация проекта

- [x] **Streamlit**: Имеется демонстрация с помощью Streamlit, которую нужно запускать вручную.
- [] **Streamlit проект размещен в интернете**: Streamlit-проект НЕ размещен в интернете, но спокойно открывается локально

### Машинное обучение

- [x] **Построение предсказательных моделей**: Построение предсказательных моделей типа регрессий или решающих деревьев.

### Дополнительные технологии

- [x] **Использование дополнительных библиотек**: Использовались на уровне «взять пример из документации и адаптировать для своих нужд».

---

## Установка зависимостей

```sh
pip install geopandas geopy pandas numpy requests seaborn matplotlib folium networkx time BeautifulSoup4 selenium scikit-learn lightgbm catboost statsmodels duckdb warnings streamlit textblob
```
---

## Запуск проекта
Для запуска проекта выполните команду:

```sh
streamlit run final_project.ipynb
```

## Контактная информация
Если у вас есть вопросы или предложения, пожалуйста, свяжитесь со мной по электронной почте: [dgolovinov@nes.ru]
