# naive_bayes.py
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load
with open("data_splits.pkl", "rb") as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

# Train
nb = MultinomialNB(alpha=0.5)
nb.fit(X_train, y_train)

# Validation
y_val_pred = nb.predict(X_val)
print("🔹 NB Validation Accuracy:", round(accuracy_score(y_val, y_val_pred), 4))

# Test
y_test_pred = nb.predict(X_test)
print("\n🔹 NB TEST RESULTS")
print(classification_report(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))