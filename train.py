from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib 

#1. Load the data (150 flowers, 4 measurements each)
X, y = load_iris(return_X_y = True)
print("Data shape:", X.shape)

#2. Split: 80% for training, 20% for testing 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

#3. Create and train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

#4. Check how good it is on data it has never seen
predictions = model.predict(X_test)
print("Test accuracy:", accuracy_score(y_test, predictions))

#5. Save the trained model to a file(we use it in Project 2)
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")

