import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from textblob import Word
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# announcement
# Here, I did the following preprocessing on the data.
# First, I cleaned the text, removed stop words, merged, tokenized and restored the word types,
# and then created a word frequency data frame.
# I changed all the three sentiments to 0,1,2 which could be better for training.
# 0 is negative,1 is neutral and 2 is positive.
# Then I also did data visualization to facilitate checking the data.(wordcloud,word frequency and sentiment distribution.)
# The last processed csv is called financial_phrase_bank_final.csv, you can use it to train the model
# if there is any problems please get in touch with Moyu or Yuan.

nltk.download('stopwords')
nltk.download('wordnet')

sw = set(stopwords.words('english'))
nw = {'no', 'nor', 'not'}
words = sw - nw


# clean text
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text


# remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in words])


# combine all strings into one sentence
def get_word_string(texts):
    text = ' '.join(texts)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.lower()
    return text


# tokenize sentence into words
def get_word(sentence):
    return nltk.RegexpTokenizer(r'\w+').tokenize(sentence)


# lemmatize words
def lemmatize_word(words):
    return [Word(word).lemmatize() for word in words]


# create frequency dataframe
def create_freq_df(words):
    word_freq = Counter(words)
    freq_df = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])
    return freq_df


def preprocess_FPB():
    path = "./financial_phrase_bank.csv"
    df = pd.read_csv(path)

    # drop NaN
    df = df.dropna(subset=['text', 'label'])

    # Apply cleaning and stopword removal
    df['text'] = df['text'].apply(clean_text)
    df['text'] = df['text'].apply(remove_stopwords)

    # Combine all cleaned texts into one sentence
    all_text = get_word_string(df['text'])

    # Tokenize the combined text
    t_words = get_word(all_text)

    # Lemmatize the words
    lemmatized_words = lemmatize_word(t_words)

    # Create frequency dataframe
    freq_df = create_freq_df(lemmatized_words)

    # Map sentiments to numbers called label
    labels = {'negative': 1, 'neutral': 2, 'positive': 0}
    df['label'] = df['label'].map(labels)

    # Save the financial phrasebank final csv
    df.to_csv("financial_phrase_bank_final.csv", index=False)

    # Save word frequency to csv
    freq_df.to_csv("word_frequency.csv", index=False)

    return df, freq_df, lemmatized_words


# Visualization is here, you can check the word frequency
def show_word_frequency(freq_df, top_n=30):
    plt.figure(figsize=(12, 8))
    top_words = freq_df.nlargest(top_n, 'Frequency')
    sns.barplot(y='Frequency', x='Word', data=top_words, hue='Word', palette='viridis', dodge=False, legend=False)
    plt.title(f'Top {top_n} Words by Frequency', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency', fontsize=14, fontweight='bold')
    plt.xlabel('Word', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    sns.despine(left=True, bottom=True)
    plt.show()


def show_sentiment_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, hue='label', palette='viridis', legend=False)
    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(ticks=[0, 1, 2], labels=['Negative', 'Neutral', 'Positive'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def show_word_cloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud')
    plt.show()


def main():
    df, freq_df, lemmatized_words = preprocess_FPB()

    # Optional: Uncomment the lines below to visualize the data
    show_word_frequency(freq_df, top_n=30)
    show_sentiment_distribution(df)
    show_word_cloud(lemmatized_words)


if __name__ == "__main__":
    main()
