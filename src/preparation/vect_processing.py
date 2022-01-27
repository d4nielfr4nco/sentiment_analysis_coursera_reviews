from sklearn.feature_extraction.text import TfidfVectorizer


class DataFrameTextVectorizer:

    def __init__(self, max_df=0.9, min_df=3, num_features=1000000):
        self.tfidf_vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=num_features)

    def vectorize_corpus(self, df, text_column):
        corpus = df[text_column].tolist()
        return self.tfidf_vectorizer.fit_transform(corpus)

    def vectorize_text(self, text):
        return self.tfidf_vectorizer.transform([text])

    def vectorize_df_text(self, df, text_column):
        texts = df[text_column].tolist()
        return self.tfidf_vectorizer.transform(texts)

    def get_vocabulary(self):
        return self.tfidf_vectorizer.vocabulary

    def get_stop_words(self):
        return self.tfidf_vectorizer.stop_words
