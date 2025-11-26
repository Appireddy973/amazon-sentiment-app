from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import re, string, nltk, io, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from werkzeug.utils import secure_filename

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

latest_df = None
model_report = ""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub("\\s+", " ", text)
    return " ".join([word for word in text.split() if word not in stop_words])

def label_sentiment(rating):
    if rating is None:
        return "Neutral"
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

def classify_review_type(text):
    service_keywords = ["delivery", "packaging", "customer service", "refund", "return", "shipping"]
    return "Service" if any(word in text.lower() for word in service_keywords) else "Product"

def convert_to_review_url(url):
    match = re.search(r'/dp/([A-Z0-9]{10})|product-reviews/([A-Z0-9]{10})', url)
    asin = match.group(1) if match and match.group(1) else match.group(2) if match else None
    if asin:
        return f"https://www.amazon.com/product-reviews/{asin}/"
    return url

def scrape_amazon_reviews(url):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    time.sleep(6)

    reviews, ratings = [], []
    review_elements = driver.find_elements(By.CSS_SELECTOR, "div[data-hook='review']")
    for element in review_elements:
        try:
            review_text = element.find_element(By.CSS_SELECTOR, "span[data-hook='review-body']").text
            rating_text = element.find_element(By.CSS_SELECTOR, "i[data-hook='review-star-rating']").text
            rating = float(rating_text.split()[0])
            reviews.append(review_text)
            ratings.append(rating)
        except:
            continue

    driver.quit()

    if not reviews:
        return pd.DataFrame()

    df = pd.DataFrame({"Review": reviews, "Rating": ratings})
    df["Cleaned"] = df["Review"].astype(str).apply(clean_text)
    df = df[df["Cleaned"].str.strip() != ""]
    df["Sentiment"] = df["Rating"].apply(label_sentiment)
    df["Review_Type"] = df["Review"].astype(str).apply(classify_review_type)
    return df

@app.route("/", methods=["GET", "POST"])
def home():
    global latest_df, model_report
    if request.method == "POST":
        if 'product_url' in request.form and request.form.get("product_url"):
            product_url = convert_to_review_url(request.form.get("product_url"))
            df = scrape_amazon_reviews(product_url)
        elif 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file:
                filename = secure_filename(uploaded_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                uploaded_file.save(filepath)
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath)
                else:
                    return render_template("index.html", error="Unsupported file type. Upload .csv or .xlsx")
                if "Review" not in df.columns:
                    return render_template("index.html", error="File must contain a 'Review' column.")
                df["Rating"] = df.get("Rating")
                df["Cleaned"] = df["Review"].astype(str).apply(clean_text)
                df = df[df["Cleaned"].str.strip() != ""]
                df["Sentiment"] = df["Rating"].apply(label_sentiment)
                df["Review_Type"] = df["Review"].astype(str).apply(classify_review_type)
            else:
                return render_template("index.html", error="No file uploaded.")
        else:
            return render_template("index.html", error="Please provide a product URL or upload a file.")

        if df.empty:
            return render_template("index.html", error="No reviews found or failed to scrape.")

        # Vectorize and split the data
        vectorizer = TfidfVectorizer()
        X = df["Cleaned"]
        y = df["Sentiment"]

        try:
            X_vec = vectorizer.fit_transform(X)
        except ValueError:
            return render_template("index.html", error="Vectorization failed. Try different data.")

        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)
        model = MultinomialNB(alpha=0.5)
        model.fit(X_train, y_train)

        # Predict only on test data for metrics
        y_pred = model.predict(X_test)
        model_report = classification_report(y_test, y_pred, digits=3)

        # Save predictions to full df for charts & download
        df["Predicted"] = model.predict(X_vec)
        latest_df = df.copy()

        # Save confusion matrix
        os.makedirs("static", exist_ok=True)
        cm = confusion_matrix(y_test, y_pred, labels=["Positive", "Neutral", "Negative"])
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Positive", "Neutral", "Negative"],
                    yticklabels=["Positive", "Neutral", "Negative"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("static/conf_matrix.png")
        plt.close()

        return render_template("result.html")
    return render_template("index.html")

@app.route('/get_chart_data')
def get_chart_data():
    global latest_df
    if latest_df is None:
        return jsonify({"error": "No data"})

    sentiment = latest_df.groupby(["Review_Type", "Sentiment"]).size().unstack(fill_value=0)
    grouped_data = []
    for review_type in sentiment.index:
        grouped_data.append({
            "type": review_type,
            "Positive": int(sentiment.loc[review_type].get("Positive", 0)),
            "Negative": int(sentiment.loc[review_type].get("Negative", 0)),
            "Neutral": int(sentiment.loc[review_type].get("Neutral", 0)),
        })

    return jsonify({
        "sentiment": latest_df["Sentiment"].value_counts().to_dict(),
        "review_type": latest_df["Review_Type"].value_counts().to_dict(),
        "grouped": grouped_data
    })

@app.route("/metrics")
def metrics():
    global model_report
    return f"<pre>{model_report}</pre><br><img src='/static/conf_matrix.png' alt='Confusion Matrix' style='max-width: 100%;'>"

@app.route("/download_metrics")
def download_metrics():
    global model_report
    return send_file(io.BytesIO(model_report.encode()), download_name="model_report.txt", as_attachment=True)

@app.route("/download")
def download():
    global latest_df
    if latest_df is not None:
        csv = latest_df.to_csv(index=False)
        return send_file(io.BytesIO(csv.encode()), download_name="sentiment_reviews.csv", as_attachment=True)
    return "No data to download."

if __name__ == "__main__":
    app.run(debug=True)
