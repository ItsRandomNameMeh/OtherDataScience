import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. Загрузка данных
df = pd.read_csv('Titanic-Dataset.csv')
print("# 1. Загрузка данных из Titanic-Dataset.csv")

# 2. Разведочный анализ данных (EDA)
print("\n# 2. Разведочный анализ данных (EDA)")
print(f'DataFrame shape: {df.shape}') # Вывод количества строк и столбцов
print(f'DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB') # Вывод объема памяти, занимаемого DataFrame
print(df.describe()) # Вывод статистических данных интервальных переменных

for col in df.select_dtypes(include='object'): # Вывод моды и количества ее встречаемости для категориальных переменных
    print(f'{col}:')
    print(f'  Mode: {df[col].mode()[0]}')
    print(f'  Mode count: {df[col].value_counts()[df[col].mode()[0]]}')

# 3. Подготовка данных
print("\n# 3. Подготовка данных")
# 3.A Обработка пропусков
print("# 3.A Обработка пропусков: заполнение медианой для Age и модой для Embarked")
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3.B Обработка выбросов (визуализация)
print("# 3.B Обработка выбросов: визуализация Fare с помощью boxplot")
sns.boxplot(x=df['Fare'])
plt.show()

# 3.C Кодирование категориальных переменных
print("# 3.C Кодирование категориальных переменных: one-hot encoding для Sex, Embarked, Pclass")
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)

# 3.D Разделение на train и test
print("# 3.D Разделение данных на train и test")
X = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Построение моделей
print("\n# 4. Построение моделей")
# 4.A KNN
print("# 4.A KNN: обучение модели KNeighborsClassifier")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# 4.B Логистическая регрессия
print("# 4.B Логистическая регрессия: обучение модели LogisticRegression")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# 4.C SVM
print("# 4.C SVM: обучение модели SVC")
svm = SVC()
svm.fit(X_train, y_train)

# 5. Оценка качества
print("\n# 5. Оценка качества моделей")
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}') # Вывод accuracy
    print(f'Precision: {precision_score(y_test, y_pred)}') # Вывод precision
    print(f'Recall: {recall_score(y_test, y_pred)}') # Вывод recall
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}') # Вывод матрицы ошибок
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f'ROC AUC: {roc_auc_score(y_test, y_prob)}') # Вывод ROC AUC
    except AttributeError:
        print('ROC AUC: not available') # Вывод сообщения, если ROC AUC недоступен

    print('-' * 20)

print("# 5.A Оценка KNN")
evaluate_model(knn, X_test, y_test)
print("# 5.B Оценка логистической регрессии")
evaluate_model(lr, X_test, y_test)
print("# 5.C Оценка SVM")
evaluate_model(svm, X_test, y_test)