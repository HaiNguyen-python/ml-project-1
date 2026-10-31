#tests/test_model.py - quality checks for the trained model 
from pathlib import Path 
import joblib 
from sklearn.datasets import load_iris 
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split

#model.joblib is in the project folder, one level above "tests"
MODEL_PATH = Path(__file__).parent.parent / "model.joblib"

#The quality gate: the model must be at least this accurate 
MIN_ACCURACY = 0.90

def load_test_data(): 
    #Use exactly the same split as train.py
    #so we test on data the model never saw
    X,y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_test, y_test 

#The Accuracy Test
def test_model_accuracy_is_high_enough():
    model = joblib.load(MODEL_PATH)
    X_test,y_test = load_test_data()
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.3f}")
    assert accuracy >= MIN_ACCURACY, f"Accuracy {accuracy:.3f} is below {MIN_ACCURACY}"

#A second check, does it predict all 3 species? 
def test_model_predicts_all_three_species():
    #A broken model often predicts the same class for everything
    model = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_data()
    predicted_species = set(model.predict(X_test)) #set removes duplicates
    assert predicted_species == {0, 1, 2}