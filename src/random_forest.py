# random_forest.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load
with open("data_splits.pkl", "rb") as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

# Train
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced", 
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Validation
y_val_pred = rf.predict(X_val)
print("🔹 RF Validation Accuracy:", round(accuracy_score(y_val, y_val_pred), 4))

# Test
y_test_pred = rf.predict(X_test)
print("\n🔹 RF TEST RESULTS")
print(classification_report(y_test, y_test_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
