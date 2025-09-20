import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Load datasets
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# 2. Add labels
df_fake['label'] = "FAKE"
df_real['label'] = "REAL"

# 3. Combine datasets
df = pd.concat([df_fake, df_real])

# 4. Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Split features and labels
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Build Pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7, ngram_range=(1,3), max_features=5000)),
    ("model", LogisticRegression(max_iter=1000))
])

# 7. Train Pipeline
pipeline.fit(X_train, y_train)

# 8. Predictions
y_pred = pipeline.predict(X_test)

# 9. Evaluation
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Precision:", precision_score(y_test, y_pred, pos_label="REAL"))
#print("Recall:", recall_score(y_test, y_pred, pos_label="REAL"))

# 10. Try a custom news input
sample_news = [
    "Border confrontations between Russia and Ukraine felt at times like flashbacks to the Cold War. After protests against Ukrainian President Viktor Yanukovych led to him fleeing the country, Russian-backed fighters moved into the seaside city of Crimea in the ensuing power vacuum."
]
print("Custom prediction:", pipeline.predict(sample_news)[0])