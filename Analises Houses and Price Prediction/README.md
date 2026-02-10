# Aнализ базы данных домов и построение модели регрессии для предсказания цены дома по параметрам

# План:

- Анализ выбросов;
- Анализ отсутствующих данных;
- Восстановление значений
- Преобразование категориальных признаков
- Создание модели линейной регрессии
- Проведение поиска отимальных параметров по сетке
- Обучение и оценка модели

``` Стек / основные инструменты анализа
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


# ML инструменты

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error, mean_squared_error
```

# Лучшая метрика:  MAE / Средняя цена ~93.5%

