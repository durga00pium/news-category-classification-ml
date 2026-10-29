import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_project():
    print("🔄 Step 1: Downloading real news dataset...")
    categories = ['rec.sport.baseball', 'sci.space', 'talk.politics.mideast']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    df_real = pd.DataFrame({'news_text': newsgroups.data, 'category': newsgroups.target})
    df_real = df_real[df_real['news_text'].str.strip() != ""]
    category_map = {0: 'Sports (Baseball)', 1: 'Tech/Space', 2: 'Business/Politics'}
    df_real['category_name'] = df_real['category'].map(category_map)

    print("🔄 Step 2: Splitting into Training (80%) and Testing (20%)...")
    y_test_data, X_test_data, y_train, y_test = train_test_split(
        df_real['news_text'], 
        df_real['category_name'], 
        test_size=0.20, 
        random_state=42
    )

    print("🔄 Step 3: Converting words to numerical vectors (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_numeric = vectorizer.fit_transform(y_test_data)
    X_test_numeric = vectorizer.transform(X_test_data)

    print("🔄 Step 4: Training the Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_numeric, y_train)

    print("🔄 Step 5: Generating predictions on test data...")
    predictions = model.predict(X_test_numeric)

    print(f"\n✅ Final Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_project()
