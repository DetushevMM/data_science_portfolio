# Мы создадим автоматический предиктор одобрения кредитных карт с использованием методов машинного обучения, как это делают настоящие банки.

# Задача: 
- Выполнить исследование и отбор признаков.
- Обучить алгоритм логистической регрессии для классификации значений.
- Выполнить подбор гипепараметров.
- Выбрать лучшую модель.
- Выполнить прогноз на проверочных данных лучшей моделью.
- Снять метрики и ошибки модели.
- Сделать выводы.

Данные "13_credit_approve" взяты здесь: https://archive.ics.uci.edu/dataset/27/credit+approval

Этот файл касается предложений кредитных карт. Все имена и значения атрибутов были изменены на бессмысленные символы для защиты конфиденциальности данных. Целевая переменная А16-«+/-».

Этот набор данных интересен тем, что в нем есть хорошее сочетание атрибутов — непрерывных, номинальных с небольшим количеством значений, и номинальных - с большим количеством значений. Также есть несколько пропущенных значений.

# Стек / основные инструменты анализа:
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns


# ML инструменты:
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Оценка модели: 
Logistic Regression - F1 score = 0.85.

# Выводы: 
Логистическая регрессия демонстрирует высокие показатели по всем метрикам, особенно по Accuracy, Recall и F1-score, что делает её более сбалансированной и эффективной для этой задачи, по сравнению с моделью деревьев решений (DecisionTreeClassifier).
