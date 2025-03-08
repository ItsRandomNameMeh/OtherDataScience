import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sqlite3
import numpy as np

# 1. Загрузка данных из FirstLab.db
print("# 1. Загрузка данных из FirstLab.db")
conn = sqlite3.connect('FirstLab.db')
query = """
SELECT film.film_id, film.rating, film.rental_rate, film.length, film.replacement_cost, film.rental_duration
FROM film
"""
df = pd.read_sql_query(query, conn)
conn.close()


# 2. Подготовка данных
# 2.A Объединение рейтингов
df = df[df['rating'] != 'G']  # Удаление фильмов с рейтингом 'G'
df['rating'] = df['rating'].replace({'PG-13': 'PG', 'NC-17': 'R'})  # Объединение рейтингов

# 2.B Преобразование рейтинга в числовой формат
rating_mapping = {'PG': 0, 'R': 1}
df['rating_numeric'] = df['rating'].map(rating_mapping)

# 2. Разведочный анализ данных (EDA)
print("\n# 3. Разведочный анализ данных (EDA)")
print(f'DataFrame shape: {df.shape}')
print(f'DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB')
print(df['rating'].value_counts()) # Вывод распределения рейтингов

# Создание матрицы корреляции
correlation_matrix = df[['rating_numeric', 'rental_rate', 'length', 'replacement_cost', 'rental_duration']].corr()

# Визуализация матрицы корреляции с помощью тепловой карты
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Матрица корреляции')
plt.show()

# 3. Подготовка данных (ПРОДОЛЖЕНИЕ)
# 3.A Кодирование целевой переменной (rating)
df = pd.get_dummies(df, columns=['rating'])

# 3.C Разделение на train и test
X = df[['rental_rate', 'length']]  # Используем выбранные параметры
y_columns = [col for col in df.columns if col.startswith('rating_')]
y = df[y_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование y в одномерный массив
y_train_single = np.argmax(y_train.values, axis=1)
y_test_single = np.argmax(y_test.values, axis=1)

# 4. Построение моделей
print("\n# 4. Построение моделей")
# 4.A KNN
print("# 4.A KNN: обучение модели KNeighborsClassifier")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train_single)

# 4.B Логистическая регрессия
print("# 4.B Логистическая регрессия: обучение модели LogisticRegression")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train_single)

# 4.C SVM
print("# 4.C SVM: обучение модели SVC")
svm = SVC()
svm.fit(X_train, y_train_single)

# 5. Оценка качества
print("\n# 5. Оценка качества моделей")
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('-' * 20)

print("# 5.A Оценка KNN")
evaluate_model(knn, X_test, y_test_single)
print("# 5.B Оценка логистической регрессии")
evaluate_model(lr, X_test, y_test_single)
print("# 5.C Оценка SVM")
evaluate_model(svm, X_test, y_test_single)