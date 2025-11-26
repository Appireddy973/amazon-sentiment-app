import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def extract_features(text):
    features = {
        "length": len(text),
        "num_exclamations": text.count("!"),
        "num_questions": text.count("?"),
        "num_uppercase": sum(1 for c in text if c.isupper()),
    }
    return features
from sklearn.feature_extraction.text import CountVectorizer

def vectorize_text(corpus):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
from sklearn.model_selection import train_test_split

     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42
    )
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Report:\n", classification_report(y_test, y_pred))
def predict_sentiment(text, model, vectorizer):
    X_input = vectorizer.transform([text])
    prediction = model.predict(X_input)
    return prediction[0]
import matplotlib.pyplot as plt

def plot_sentiment_distribution(predictions):
    plt.hist(predictions, bins=3, edgecolor='black')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Frequency")
    plt.show()
import pandas as pd

    def load_data(file_path):
            df = pd.read_csv(file_path)
            return df["review"], df["sentiment"]
            import seaborn as sns

            def plot_word_counts(corpus):
                word_counts = pd.Series(" ".join(corpus).split()).value_counts().head(20)
                sns.barplot(x=word_counts.values, y=word_counts.index)
                plt.title("Top 20 Most Common Words")
                plt.xlabel("Frequency")
                plt.ylabel("Words")
                plt.show()
                def clean_text(text):
                text = text.lower()
                text = re.sub(r"[^a-z\s]", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                return text
                                       