# Определенить твёрдость горной породы для выбора нужной насадки на проходческий щит

Данные для задачи регрессии: Смоделированные данные соотношения рентгеновского сигнала и плотности скальной породы.

Заказчику нужно пробурить туннель в скале с использованием туннеле-проходческих комплексов. Спереди устройства находится щит с различными насадками, в зависимости от твердости скальной породы. Для определения твердости породы используются рентгеновские лучи. Силу отражения сигнала в nHz преобразуют в оценку плотности скальной породы в кг/куб.м.

# Задача: 
на основе лабораторных экспериментов для материалов различной плотности определенить твёрдость горной породы для выбора нужно насадки на щит.
У нас всего 2 колонки: уровень отраженного сигнала и плотность породы. Переменуем эти колонки.
Модель машинного обучения должна принимать сигнал в nHz и выдавать плотность горной породы в кг/куб.м.

# В результате решения этой задачи мы сравним разные модели:

- Линейная регрессия;
- Полиномиальная регрессия;
- Регрессия KNN;
- Регрессия деревьев решений (Decision Trees);
- Регрессия методом опорных векторов (CVR);
- Регрессия расширяемых деревьев (boosted trees);
- Регрессия случайных лесов (Random Forest).

# Результат:
Наша финальная модель - это SVR. 
- MAE: 0.108
- RMSE: 0.126
```
# Стек / основные инструменты

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML инструменты

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
