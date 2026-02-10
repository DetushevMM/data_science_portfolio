# Изучим отток клиентов (churn) для провайдера интерент услуг и телефонаии.

Что может сделать компания для уменьшения оттока в месяц и в год? Для контрактов в месяц люди могли заранее решить, что будут пользрваться услушами не долго.

# План
- Проведем исследование данных (exploratory data analysis - EDA)
- Когортный анализ оттока клиентов (churn cohort analysis). Когда клиент перестает пользоваться услугами компании.
- Предиктивное машинное обучение. На основе признаков определим, что с высокой вероятностью клиент уйдет в отток.

Мы исследуем 4 модели на основе деревьев: одно дерево решений, случайный лес, адаптивный бустинг и градиентный бустинг.

Oсновные характеристики датафрейма:
 -  gender             
 -  SeniorCitizen     
 -  Partner           
 -  Dependents        
 -  tenure             
 -  PhoneService      
 -  MultipleLines     
 -  InternetService   
 -  OnlineSecurity  
 -  OnlineBackup   
 -  DeviceProtection  
 -  TechSupport     
 -  StreamingTV   
 -  StreamingMovies  
 -  Contract       
 -  PaperlessBilling  
 -  PaymentMethod     
 -  MonthlyCharges    
 -  TotalCharges      
 -  Churn     

# Результат
Выбрана модель AdaBoost -	Recall = 0.91.

В данной задаче следует рассмотреть другие методы, например опорных векторов или логистическую регрессию или провести работу с данными совместно с экспертом (feature engineering)

# Стек / основные инструменты
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# ML инструменты
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from sklearn.tree import plot_tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier


