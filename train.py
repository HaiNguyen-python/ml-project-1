from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model.joblib"

#1. Load the data (150 flowers, 4 measurements each)
X, y = load_iris(return_X_y = True)
print("Data shape:", X.shape)

#2. Split: 80% for training, 20% for testing 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#3. Create and train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

for name, m in [
    ("Logistic Regression", LogisticRegression(max_iter=200)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42)),
]:
    scores = cross_val_score(m, X, y, cv=5) #5 different train/test splits
    print(f"{name}: mean accuracy = {scores.mean():.3f}")

#4. Check how good it is on data it has never seen
predictions = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, predictions))

#5. Save the trained model to a file(we use it in Project 2)
joblib.dump(model, MODEL_PATH)
print("Model saved to", MODEL_PATH)

