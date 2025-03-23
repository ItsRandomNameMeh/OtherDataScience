import pandas as pd
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import TSNE
import joblib
import numpy as np


def main():
    db_file = 'database.sqlite'  # Замените на путь к вашему файлу базы данных

    try:
        conn = sqlite3.connect(db_file)
        query = "SELECT Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species FROM Iris"
        df = pd.read_sql_query(query, conn)
        conn.close()

        numeric_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        for col in numeric_cols:
            z_scores = stats.zscore(df[col])
            df = df[(z_scores < 3) & (z_scores > -3)]

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        plt.figure(figsize=(15, 10))  # Создаем фигуру для всех графиков
        for i, kernel in enumerate(kernels):
            kpca = KernelPCA(n_components=2, kernel=kernel)
            df_kpca = kpca.fit_transform(df[numeric_cols])

            plt.subplot(2, 3, i + 1)  # Создаем подграфик
            plt.scatter(df_kpca[:, 0], df_kpca[:, 1], c=pd.Categorical(df['Species']).codes)
            plt.title(f'Kernel PCA ({kernel})')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
        plt.tight_layout()  # Улучшаем расположение графиков
        plt.show()

        # 4. Вывод по Kernel PCA
        print("\nВыводы по Kernel PCA:")
        print("Разные ядерные функции влияют на то, как данные проецируются в двумерное пространство.")
        print(
            "Нелинейные ядра (poly, rbf, sigmoid, cosine) могут лучше разделять классы, если данные нелинейно разделимы.")
        print("Линейное ядро работает хорошо, если данные линейно разделимы.")

        # 5. Дисперсия и lost_variance для линейного ядра
        kpca_linear = KernelPCA(n_components=2, kernel='linear')
        df_kpca_linear = kpca_linear.fit_transform(df[numeric_cols])

        # Для линейного ядра KernelPCA эквивалентен PCA, поэтому можно использовать PCA
        pca = PCA(n_components=2)
        pca.fit(df[numeric_cols])
        explained_variance = pca.explained_variance_ratio_
        lost_variance = 1 - np.sum(explained_variance)

        print("\nДисперсия и lost_variance для линейного ядра:")
        print(f"Объясненная дисперсия: {explained_variance}")
        print(f"Потерянная дисперсия: {lost_variance}")
        print("Потерянная дисперсия показывает, сколько информации было потеряно при снижении размерности.")

        # 6. t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        df_tsne = tsne.fit_transform(df[numeric_cols])
        plt.figure()
        plt.scatter(df_tsne[:, 0], df_tsne[:, 1], c=pd.Categorical(df['Species']).codes)
        plt.title('t-SNE')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.show()

        print("\nВыводы по t-SNE:")
        print("t-SNE - алгоритм нелинейного снижения размерности, который может лучше сохранять локальную структуру данных.")

        # 7. Выгрузка и загрузка модели
        joblib.dump(kpca_linear, 'kpca_linear_model.joblib')
        loaded_model = joblib.load('kpca_linear_model.joblib')
        df_loaded_kpca = loaded_model.transform(df[numeric_cols])
        print("\nМодель Kernel PCA с линейным ядром выгружена и загружена.")

    except sqlite3.Error as e:
        print(f"Ошибка при работе с базой данных: {e}")


if __name__ == "__main__":
    main()