# module_titanic.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def load_data(train_path, test_path):
    """
    Загрузка данных из CSV файла.
    :param train_path: Путь к CSV файлу с обучающей выборкой.
    :param test_path: Путь к CSV файлу с тестовой выборкой.
    :return: DataFrame с загруженными данными.
    """
    return pd.read_csv(train_path), pd.read_csv(test_path)

def preprocess_data(train_df, test_df):
    """
    Обработка данных перед обучением.
    :param train_df: обучающий DataFrame.
    :param test_df: тестовый DataFrame.
    :return: комбинация DataFrame'ов с обработанными данными.
    """
    # исключение неиспользуемых колонок
    features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']
    train_df = train_df.drop(features_drop, axis=1)
    test_df = test_df.drop(features_drop, axis=1)
    train_df = train_df.drop('PassengerId', axis=1)
    
    train_test_data = [train_df, test_df] # combine data

    # колонка Sex: замена на значения 0/1
    for dataset in train_test_data:
        dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # колонка Age: замена значений null произвольным числом от (mean_age - std_age) до (mean_age + std_age) 
    for dataset in train_test_data:
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
        dataset['Age'] = dataset['Age'].astype(int)
    
    # колонка Age: преобразование в диапазоны
    for dataset in train_test_data:
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age'] = 4    

    # колонка Fare: установка средних значений вместо пустых
    for dataset in train_test_data:
        dataset['Fare'] = dataset['Fare'].fillna(train_df['Fare'].median())    

    # колонка Fare: преобразование в диапазоны
    for dataset in train_test_data:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    return train_df, test_df

def show_stat(df):
    # отображение основной статистики
    # числовые значения
    print(df.describe())
    # объектные значения
    print(df.describe(include=['O']))   
    # график
    sns.barplot(x='Sex', y='Survived', data=df)

def train_logistic_regression(X_train, y_train, X_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred_log_reg = clf.predict(X_test)
    acc_log_reg = round(clf.score(X_train, y_train) * 100, 2)
    print('Logistic regression result:')
    print(str(acc_log_reg) + ' percent')    

def train_decision_tree(X_train, y_train, X_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred_decision_tree = clf.predict(X_test)
    acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
    print('Decision tree result:')
    print(str(acc_decision_tree) + ' percent')    
