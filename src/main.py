import subprocess
import sys

def run_step(description, command):
    print("\n" + "="*60)
    print(f"▶ {description}")
    print("="*60)
    result = subprocess.run([sys.executable, command])
    if result.returncode != 0:
        print(f"❌ Error while running {command}")
        sys.exit(1)

if __name__ == "__main__":

    # 1️⃣ Preprocessing
    run_step(
        "Step 1: Preprocessing & Cleaning Data",
        "preprocess.py"
    )

    # 2️⃣ Feature Extraction + Split
    run_step(
        "Step 2: Feature Engineering & Data Split",
        "features_balancing.py"
    )

    # 3️⃣ Naive Bayes
    run_step(
        "Step 3: Training & Evaluating Naive Bayes",
        "naive_bayes.py"
    )

    # 4️⃣ Random Forest
    run_step(
        "Step 4: Training & Evaluating Random Forest",
        "random_forest.py"
    )

    # 5️⃣ Neural Network
    run_step(
        "Step 5: Training & Evaluating Neural Network (MLP)",
        "neural_network.py"
    )

    print("\n" + "="*60)
    print("✅ FULL PIPELINE EXECUTED SUCCESSFULLY")
    print("="*60)
