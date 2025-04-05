import re
from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger, Doc
from nltk.corpus import stopwords, words
import nltk, ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('words')

# Загрузка русских стоп-слов
nltk.download('stopwords')
nltk.download('words')
russian_stopwords = set(stopwords.words('russian'))
english_stopwords = set(stopwords.words('english'))
english_vocab = set(words.words())

# Наташа: инициализация
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)

def preprocess_text(text):
    # Удаление знаков препинания и цифр
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Лемматизация с помощью Natasha
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    # Лемматизируем и удаляем стоп-слова
    lemmas = []
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        lemma = token.lemma
        if (
            lemma and
            lemma not in russian_stopwords and
            lemma not in english_stopwords and
            lemma not in english_vocab and
            len(lemma) > 2
        ):
            lemmas.append(lemma)

    return ' '.join(lemmas)

def main():
    # Чтение исходного файла с песнями
    with open('songs_dataset.txt', 'r', encoding='utf-8') as file:
        raw_text = file.read()

    # Разделение текстов песен по разделителю
    songs = raw_text.split('---')

    processed_songs = []
    for idx, song in enumerate(songs, start=1):
        song = song.strip()
        if song:
            cleaned_song = preprocess_text(song)
            processed_songs.append(cleaned_song)
            print(f"Песня {idx} обработана")

    # Сохраняем обработанные песни в новый файл
    with open('songs_dataset_cleaned.txt', 'w', encoding='utf-8') as file:
        for song in processed_songs:
            file.write(song + '\n\n---\n\n')

    print("Все песни успешно обработаны и сохранены в songs_dataset_cleaned.txt")

if __name__ == '__main__':
    main()
