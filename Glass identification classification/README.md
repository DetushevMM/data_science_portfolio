# Построим модель классификации стекла по химическому составу.

# План:

- EDA: Анализируем данные, проверяем на пропуски, визуализируем связи.
- Предобработка: Масштабируем данные, разделяем их на обучающую и тестовую выборки.
- Модели: Обучаем три классификационные модели — Logistic Regression, Random Forest и SVM.
- Подбор гиперпараметров: Для каждой модели используем GridSearchCV для нахождения оптимальных гиперпараметров.
- Оценка: Оценим модели по меткам точности, F1-скор и матрицам ошибок.
- Сравнение: Сравниваем точности моделей.
- Сохранение: Сохраняем лучшую модель для дальнейшего использования.

Для классификации мы будем использовать три модели: 
- Logistic Regression,
- Random Forest
- Support Vector Machine (SVM).

# Результат:
Для сравнения моделей можно использовать их точность или F1-score. 
- Точность Logistic Regression: 0.627906976744186
- Точность Random Forest: 0.8372093023255814 - лучшая модель
- Точность SVM: 0.7209302325581395
```
# Стек / основные иструменты анализа

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# ML инструменты

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
