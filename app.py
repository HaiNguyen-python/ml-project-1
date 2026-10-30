#app.py - a web API that serves the trained Iris model
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

#Find model.joblib next to this file, wherever the terminal is
MODEL_PATH = Path(__file__).parent / "model.joblib"

#Load the trained model once, when the API starts 
model = joblib.load(MODEL_PATH)

#Names of the 3 flower species (index 0, 1, 2)
SPECIES = ["setosa", "versicolor", "virginica"]

app = FastAPI(title = "Iris Classifier API")

# Describe the input the API expects (4 measurements in cm).
# gt = greater than, le = less than or equal: values outside this range get a 422 error
class Flower(BaseModel):
    sepal_length: float = Field(gt=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(gt=0, le=10, description="Petal length in cm")
    petal_width: float = Field(gt=0, le=10, description="Petal width in cm")

@app.get("/")
def home():
    #Friendly message for the root address
    return {"message": "Iris Classifier API. Open /docs to try it."}

@app.get("/health")
def health():
    #Simple check that the API is running: 
    return {"status": "ok"}

@app.post("/predict")
def predict(flower: Flower):
    #Put the 4 values in the order the model was trained on: 
    features = [[
        flower.sepal_length,
        flower.sepal_width,
        flower.petal_length,
        flower.petal_width,
    ]]
    prediction = model.predict(features)[0]
    return {"species": SPECIES[prediction]}

# Start the web server when this file is run directly (e.g. with the Run button)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)