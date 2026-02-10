# Построим модель для обнаружения мошеннических транзакций по кредитным картам на основе имеющихся признаков.

# Задача 
- Сравнение моделей Логистическая регрессия, Random Forest, XGBoost, LightGBM и CatBoost, 
- Сохранение модели при момощи pipeline

# Описание датасета:
Данный датасет содержит информацию о транзакциях по кредитным картам, совершённых в течение двух дней в сентябре 2013 года держателями карт в Европе. Он включает 284 807 транзакций, из которых 492 являются мошенническими (около 0.17%).

Колонки V1–V28 — анонимизированные числовые признаки, полученные с помощью метода главных компонент (PCA), чтобы скрыть исходные данные.

Time — количество секунд, прошедших с момента первой транзакции в выборке.

Amount — сумма транзакции.

Class — целевая переменная:

0 — нормальная (не мошенническая) транзакция

1 — мошенническая транзакция

Данные взяты тут: https://www.kaggle.com/mlg-ulb/creditcardfraud

Особое внимание требуется уделить:
- Обработке несбалансированных классов, так как количество мошеннических транзакций крайне мало.
- Выбору и оценке моделей, способных эффективно выявлять редкие, но критически важные случаи.
```
# Стек / основные инструменты анализа
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


# ML инструменты
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import RandomizedSearchCV
```
# Вывод:

- Если приоритет — точность (меньше ложных тревог): ➜ Random Forest
- Если приоритет — recall (поймать больше мошенников): ➜ CatBoost
- Если нужен баланс с высоким AUC: ➜ XGBoost
- XGBoost — наиболее сбалансированный вариант. Он сохраняет высокий recall, не теряя слишком много precision, и показывает лучший ROC AUC, что говорит о высокой способности к различению классов. 

# Оценка модели XGBoost - ROC-AUC = 0.98.
