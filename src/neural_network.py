# neural_network.py
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load
with open("data_splits.pkl", "rb") as f:
    X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

# Train
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0001,
    max_iter=50, 
    random_state=42
)
mlp.fit(X_train, y_train)

# Save trained Neural Network model
with open("mlp_model.pkl", "wb") as f:
    pickle.dump(mlp, f)

print("✅ Neural Network model saved as mlp_model.pkl")

# Validation
y_val_pred = mlp.predict(X_val)
print("🔹 MLP Validation Accuracy:", round(accuracy_score(y_val, y_val_pred), 4))

# Test
y_test_pred = mlp.predict(X_test)
print("\n🔹 MLP TEST RESULTS")
print(classification_report(y_test, y_test_pred))


print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
