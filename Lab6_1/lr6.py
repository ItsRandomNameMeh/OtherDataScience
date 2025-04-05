import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import numpy as np
from gensim.models import Word2Vec
from collections import Counter

# Загрузка обработанных песен
def load_songs(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        raw_text = file.read()
    songs = [song.strip() for song in raw_text.split('---') if song.strip()]
    return songs

# TF-IDF
def calculate_tfidf(songs):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(songs)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_means = np.mean(X.toarray(), axis=0)
    top_indices = tfidf_means.argsort()[::-1]
    top_words = [(feature_names[i], tfidf_means[i]) for i in top_indices[:20]]
    return top_words, dict(zip(feature_names, tfidf_means))

# WordCloud
def plot_wordcloud(word_freq):
    wordcloud = WordCloud(width=800, height=600, background_color='white')
    wordcloud.generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud по TF-IDF', fontsize=16)
    plt.show()

# Подготовка данных для Word2Vec
def prepare_data_for_w2v(songs):
    tokenized_songs = [song.split() for song in songs]
    return tokenized_songs

# Обучение модели Word2Vec
def train_word2vec(tokenized_songs):
    model = Word2Vec(sentences=tokenized_songs, vector_size=100, window=5, min_count=1, workers=4)
    return model

# Функция для показа похожих слов
def show_similar(model, word):
    if word in model.wv:
        print(f"Слова, похожие на '{word}':")
        for similar_word, similarity in model.wv.most_similar(word):
            print(f"{similar_word}: {similarity:.4f}")
    else:
        print(f"Слово '{word}' не найдено в словаре модели!")

# График t-SNE
def plot_tsne(model, top_words):
    words = [word for word, _ in top_words[:15]]
    word_vectors = np.array([model.wv[word] for word in words])

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(10, 7))
    plt.scatter(embeddings[:, 0], embeddings[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))

    plt.title('t-SNE график для 15 частых слов', fontsize=16)
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.grid(True)
    plt.show()

def main():
    # Загружаем обработанные тексты песен
    songs = load_songs('songs_dataset_cleaned.txt')

    # TF-IDF и WordCloud
    top_words, word_freq = calculate_tfidf(songs)
    print("Наиболее важные слова по TF-IDF:")
    for word, score in top_words:
        print(f"{word}: {score:.4f}")
    plot_wordcloud(word_freq)

    # Word2Vec
    tokenized_songs = prepare_data_for_w2v(songs)
    model = train_word2vec(tokenized_songs)

    # Проверка близких слов
    # example_word = top_words[0][0]  # Самое важное слово по TF-IDF
    example_word = "хотеть"
    show_similar(model, example_word)

    # Построение t-SNE графика
    # Вычисляем самые частые слова в датасете (не по TF-IDF, а по частоте встречаемости)
    all_words = [word for song in tokenized_songs for word in song]
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(15)

    plot_tsne(model, most_common_words)

if __name__ == '__main__':
    main()
