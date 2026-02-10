# Рейтинг акций Google 
Анализ доверия к акциям. Построение доверительной оценки. Применение базовой математической статистики для Data scientists.

# Задачи:

- Построить доверительный интервал для цен акций из yahoo finance;
- Выделить аномалии;
- Сделать выводы по наличию или отсутствию аномальных событий.

Источник данных - yahoo finance Проведение анализа реальных котировок (цен финансовых инструментов) настоящих акций Google на бирже. Анализ из сервиса yahoo finance на котировках за интересующий период - 2018 - 2019 годы.

# Стек / основные инструменты:
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import statsmodels.api as sm

import sys

import subprocess

from scipy.stats import kurtosis, skew, norm


