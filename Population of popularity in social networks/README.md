# Предскажем популярность по количеству акций в социальных сетях.

DataSet
https://archive.ics.uci.edu/dataset/332/online+news+popularity

Данные о публикации новостей и количество репостов в социальных сетях. Этот набор данных суммирует гетерогенный набор функций о статьях, опубликованных Mashable за два года. Цель состоит в том, чтобы предсказать количество акций в социальных сетях (популярность).

# Задача:
- Уменьшить количество анализируемых признаков, применить методы PCA и t-SNE.
- Выполнить кластеризацию используя алгоритмы k-means, c-means.
- Цель: Предсказать популярность

# План:
- Предварительный анализ данных. Проверить наличие пропусков, выбросов, дисбаланса. Анализ распределения целевой переменной (shares).
- Снижение размерности. PCA: Проверить долю объяснённой дисперсии для выбора оптимального количества компонент. t-SNE: Визуализация данных в 2D/3D для изучения структуры.
- Кластеризация. K-Means: Определить оптимальное количество кластеров (методы локтя и силуэта). C-Means (Fuzzy C-Means): Проверить размытые кластеры и их характеристики.
- Анализ результатов. Оценить качество кластеризации. Связь кластеров с популярностью статей. Выявление закономерностей, влияющих на количество репостов.

Датасет содержит 58 признаков, описывающих статьи, опубликованные на Mashable. Эти признаки включают:
- Лексические характеристики: число слов в заголовке (n_tokens_title), количество уникальных токенов (n_unique_tokens), длина токенов (average_token_length), субъективность заголовка (title_subjectivity), полярность (title_sentiment_polarity) и т. д.
- Структурные характеристики: количество ссылок (num_hrefs), количество изображений (num_imgs), количество видео (num_videos).
- Показатели эмоциональной окраски: средняя положительная (avg_positive_polarity) и отрицательная (avg_negative_polarity) тональность статьи.
- Целевая переменная (shares): количество репостов статьи в социальных сетях (популярность).


# Результат:
- Silhouette Score для кластеризации на тестовой выборке: 0.5295.
Результат в 0.5295 говорит о том, что кластеры не являются идеальными, но модель KMeans всё же нашла некоторую структуру в данных. Это может быть приемлемым результатом для некоторых типов данных, но также может свидетельствовать о том, что улучшение модели или изменение количества кластеров могло бы дать более чёткие результаты.
```
# Стек / основные инструменты

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ML инструменты 

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
