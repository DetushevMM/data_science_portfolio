# Классификация авторов текстов.	

# Задача:
Обработка текстов с помощью нейронных сетей.

Подборка авторов в датасете:
- О. Генри
- Братья Стругацкие
- М. Булгаков
- Клиффорд Саймак-
- Макс Фрай
- Рэй Брэдберри

# Результат: 
Embedding + Dense	accuracy = ~ 72%

# Стек / основные инструменты
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from keras import utils

from keras.models import Sequential

from keras.layers import Dense, Dropout, SpatialDropout1D, Embedding, Flatten, Activation, Conv1D

from keras.optimizers import Adam, RMSprop, Adadelta

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from keras.layers import BatchNormalization
