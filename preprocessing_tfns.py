import re
import pandas as pd
import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import seaborn as sns

# # ANNOUNCEMENT This dataset has been split into training and validation sets and has been processed exactly the
# same. You can use it directly. The processing is basically the same as the financial phrase bank, which is also
# visualized below, but is now disabled. the two final CSVs are for the training set, the name is
# tweet_financial_training_set_final.csv for the validation set, the name is tweet_financial_validation_set_final.csv
# if you have any questions, please contact Moyu or Yuan

# ntlk is to preprocess the text, I downloaded some of them.
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_tnfs(df):
    df = df.dropna(subset=['text', 'label'])

    def preprocessing(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        words = word_tokenize(text)
        # remove stop words
        filtered_words = []
        for word in words:
            if word.lower() not in stop_words:
                filtered_words.append(word)
        # Lemmatize words
        lemmatized_words = []
        for word in filtered_words:
            lemmatized_words.append(lemmatizer.lemmatize(word))

        return ' '.join(lemmatized_words)

    df['text'] = df['text'].apply(preprocessing)
    df = df[df['text'].str.strip().astype(bool)]
    return df


# path
def read():
    path_1 = 'zeroshot/twitter-financial-news-sentiment/sent_train.csv'
    train_df = pd.read_csv(path_1)
    train_df = preprocess_tnfs(train_df)
    train_df.to_csv("tweet_financial_training_set_final.csv", index=False)

    path_2 = 'zeroshot/twitter-financial-news-sentiment/sent_valid.csv'
    valid_df = pd.read_csv(path_2)
    valid_df = preprocess_tnfs(valid_df)
    valid_df.to_csv("tweet_financial_validation_set_final.csv", index=False)

    return train_df, valid_df


def visualization(df):
    label_map = {0: 'bearish', 1: 'bullish', 2: 'neutral'}
    df['label'] = df['label'].map(label_map)

    text = ' '.join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()

    # Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False)
    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # count vectorizer
    count_vectorizer = CountVectorizer(max_features=30)
    word_counts = count_vectorizer.fit_transform(df['text'])
    word_counts_sum = word_counts.sum(axis=0)
    vocabulary_items = count_vectorizer.vocabulary_.items()
    words_freq = []
    for word, idx in vocabulary_items:
        freq = word_counts_sum[0, idx]
        words_freq.append((word, freq))
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    words = []
    frequency = []
    for word, freq in words_freq:
        words.append(word)
        frequency.append(freq)

    plt.figure(figsize=(10, 5))
    plt.bar(words, frequency)
    plt.title('Top 30 Words Frequency')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()


def main():
    train_df, valid_df = read()

    visualization(train_df)
    visualization(valid_df)


if __name__ == "__main__":
    main()
