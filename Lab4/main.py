import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

df = 0
X_train, X_test, y_train, y_test = 0,0,0,0
def task1():
    global df
    # 1. Загрузка данных
    df = pd.read_csv("Titanic-Dataset.csv")

def task2():
    global df
    # 2. Разведочный анализ данных
    print("2. Разведочный анализ данных")

    # a. Количество строк и столбцов
    print("\na. Количество строк и столбцов:")
    print(f"Количество строк: {df.shape[0]}")
    print(f"Количество столбцов: {df.shape[1]}")

    # b. Объем занимаемой памяти
    print("\nb. Объем занимаемой памяти:")
    print(f"Объем занимаемой памяти: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # c. Статистика для интервальных переменных
    print("\nc. Статистика для интервальных переменных:")
    interval_vars = df.select_dtypes(include=['number']).columns
    for var in interval_vars:
        stats = df[var].describe(percentiles=[0.25, 0.5, 0.75])
        print(f"\nСтатистика для {var}:")
        print(f"  Мин: {stats['min']}")
        print(f"  25-й персентиль: {stats['25%']}")
        print(f"  Медиана: {stats['50%']}")
        print(f"  Среднее: {stats['mean']}")
        print(f"  75-й персентиль: {stats['75%']}")
        print(f"  Макс: {stats['max']}")

    # d. Статистика для категориальных переменных
    print("\nd. Статистика для категориальных переменных:")
    categorical_vars = df.select_dtypes(include=['object']).columns
    for var in categorical_vars:
        mode = df[var].mode()[0]
        mode_count = df[var].value_counts()[mode]
        print(f"\nСтатистика для {var}:")
        print(f"  Мода: {mode}")
        print(f"  Количество моды: {mode_count}")

def task3():
    global df,  X_train, X_test, y_train, y_test
    # 3. Подготовка датасета к построению моделей ML
    # a. Анализ и обработка пропусков
    print("\n3.a. Анализ и обработка пропусков:")
    print(df.isnull().sum())  # Вывод количества пропусков в каждом столбце

    # Обработка пропусков в столбце Age
    df['Age'] = df['Age'].fillna(df['Age'].median())  # Замена пропусков медианой

    # Обработка пропусков в столбце Embarked
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Замена пропусков модой

    # Удаление столбца Cabin из-за большого количества пропусков
    df.drop('Cabin', axis=1, inplace=True)

    # b. Анализ и обработка выбросов
    print("\n3.b. Анализ и обработка выбросов:")
    # Пример обработки выбросов в столбце Fare с помощью Z-оценки
    z_scores = np.abs(stats.zscore(df['Fare']))
    threshold = 3
    outliers = np.where(z_scores > threshold)[0]
    df.drop(outliers, inplace=True)

    # c. Анализ и обработка категориальных переменных
    print("\n3.c. Анализ и обработка категориальных переменных:")
    categorical_vars = df.select_dtypes(include=['object']).columns
    print(f"Категориальные переменные: {categorical_vars}")

    # Кодирование категориальных переменных с помощью one-hot encoding
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    # Удаление ненужных столбцов
    df.drop(['Name', 'Ticket'], axis=1, inplace=True)

    # d. Построение и проверка гипотез
    print("\n3.d. Построение и проверка гипотез:")
    # Гипотеза 1: Женщины выживали чаще мужчин
    survival_by_sex = df.groupby('Survived')['Sex_male'].value_counts(normalize=True).unstack()
    print("\nГипотеза 1: Женщины выживали чаще мужчин")
    print(survival_by_sex)

    # Гипотеза 2: Пассажиры первого класса выживали чаще
    survival_by_pclass = df.groupby('Survived')['Pclass'].value_counts(normalize=True).unstack()
    print("\nГипотеза 2: Пассажиры первого класса выживали чаще")
    print(survival_by_pclass)

    # e. Разделение датасета на трейн и тест
    print("\n3.e. Разделение датасета на трейн и тест:")
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")

def task4_5():
    global  X_train, X_test, y_train, y_test, logreg
    # 4. Обучение моделей
    # a. KNN
    print("\n4.a. KNN:")
    knn = KNeighborsClassifier(n_neighbors=5)  # Создаем модель KNN с 5 соседями
    knn.fit(X_train, y_train)  # Обучаем модель
    y_pred_knn = knn.predict(X_test)  # Делаем предсказания
    accuracy_knn = accuracy_score(y_test, y_pred_knn)  # Оцениваем точность
    print(f"Точность KNN: {accuracy_knn}")

    # b. Логистическая регрессия
    print("\n4.b. Логистическая регрессия:")
    logreg = LogisticRegression(random_state=42, max_iter=1000)  # Создаем модель логистической регрессии
    logreg.fit(X_train, y_train)  # Обучаем модель
    y_pred_logreg = logreg.predict(X_test)  # Делаем предсказания
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)  # Оцениваем точность
    print(f"Точность логистической регрессии: {accuracy_logreg}")
    # 5. Оценка качества алгоритмов
    print("\n5. Оценка качества алгоритмов:")

    # Функция для оценки качества модели
    def evaluate_model(y_true, y_pred, model_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        print(f"\nОценка качества {model_name}:")
        print(f"  Точность (Accuracy): {accuracy}")
        print(f"  Точность (Precision): {precision}")
        print(f"  Полнота (Recall): {recall}")
        print(f"  F1-мера: {f1}")
        print(f"  AUC-ROC: {auc_roc}")
        print(f"  Матрица ошибок:\n{conf_matrix}")

    # Оценка KNN
    evaluate_model(y_test, y_pred_knn, "KNN")

    # Оценка логистической регрессии
    evaluate_model(y_test, y_pred_logreg, "Логистическая регрессия")

    # Выбор оптимального алгоритма
    print("\nВыбор оптимального алгоритма:")
    if accuracy_score(y_test, y_pred_knn) > accuracy_score(y_test, y_pred_logreg):
        print("KNN показывает лучшую точность.")
    else:
        print("Логистическая регрессия показывает лучшую точность.")

def task6():
    # 6. Сохранение модели
    joblib.dump(logreg, 'myTitanicModel.joblib')  # Сохраняем модель в файл
    print("Модель сохранена в файл: myTitanicModel.joblib")

task1(),task2(),task3(),task4_5(), task6()