import nltk
nltk.download('crubadan')
nltk.download('omw-1.4')
import time
import unicodedata


class DataFrameTextProcessing:

    def __init__(self, language='english'):
        self.language = language
        self.language_classifier = nltk.classify.textcat.TextCat()
        self.stopwords = set(nltk.corpus.stopwords.words(language))
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    @staticmethod
    def is_punct(word):
        return all(
            unicodedata.category(char).startswith('P') for char in word
        )

    def filter_out_other_languages(self, df, language_column):
        return df[df[language_column] == self.language[0:3]]

    def predict_text_language(self, review):
        return self.language_classifier.guess_language(review)

    def predict_language(self, df, text_column, measure_time):
        if measure_time:
            start = time.time()
            language_column = df[text_column].apply(self.predict_text_language)
            end = time.time()
            print('It took {} seconds to predict the language of all reviews.'.format(end - start))
        else:
            language_column = df[text_column].apply(self.predict_text_language)
        return language_column

    def normalize_df(self, df, text_column):

        def pos_tagger(tag):
            tag_map = {
                'N': nltk.corpus.wordnet.NOUN,
                'V': nltk.corpus.wordnet.VERB,
                'R': nltk.corpus.wordnet.ADV,
                'J': nltk.corpus.wordnet.ADJ
            }
            return tag_map.get(tag[0], nltk.corpus.wordnet.NOUN)

        def normalize_text(text):
            lowered_text = text.lower()
            return ' '.join([
                self.lemmatizer.lemmatize(word, pos_tagger(tag))
                for sentence in nltk.tokenize.sent_tokenize(lowered_text)
                for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence))
                if word not in self.stopwords and not self.is_punct(word)
            ])

        return df[text_column].apply(normalize_text)
