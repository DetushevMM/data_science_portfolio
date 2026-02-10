# Выполним кластеризацию научных патентов с применением hierarchical clustering, DBSCAN и K-means.

Датасет содержит реферативные данные о патентах в различных областях. 

Датасет содержит 2708 записей о патентах, представленных четырьмя текстовыми столбцами:

- Application Date – дата подачи заявки на патент.
- Title – название патента.
- abstract – аннотация.
- text – основной текст патента.

# План анализа:

- Предварительная обработка данных
- Векторизация текстов
- Кластеризация патентов
- Оценка результатов
- Анализ тематической принадлежности кластеров.
- Сравнение результатов Hierarchical Clustering, K-means и DBSCAN.

# Результат:
DBSCAN K-means не справились с задачей и не смогли разделить данные на 3 кластера.

Иерархическая кластеризация справилась с задачей:
- Кластер 1 имеет наиболее выраженное смещение по PC1 (0.49).
- Кластер 2 немного ниже по PC1 и PC2, но незначительно.
- Кластер 3 — наиболее сбалансированный по первым компонентам.

# Стек / основные инструменты анализа
| import pandas as pd
| import numpy as np
| import matplotlib.pyplot as plt
| import seaborn as sns
| import re
| from mpl_toolkits.mplot3d import Axes3D

# ML инструменты

  | from sklearn.feature_extraction.text import TfidfVectorizer;
  | import scipy.cluster.hierarchy as sch;
  | from scipy.cluster.hierarchy import linkage, fcluster;
  | from sklearn.model_selection import GridSearchCV
  | from sklearn.cluster import DBSCAN
  | from sklearn.metrics import make_scorer, silhouette_score, davies_bouldin_score
  | from sklearn.cluster import KMeans
  | from sklearn.metrics import silhouette_score



