#tests/test_api.py - automated tests for the Iris API 
from fastapi.testclient import TestClient
from app import app

#A fake browser that sends requests directly to the app (no server needed)
client = TestClient(app)

def test_health():
    #The health endpoint should always answer "ok"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_setosa():
    #A typical setosa flower: small petals
    flower = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    response = client.post("/predict", json=flower)
    assert response.status_code == 200
    assert response.json() == {"species": "setosa"}

def test_predict_virginica():
    #A typical virginica flower: large petals
    flower = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
    response = client.post("/predict", json=flower)
    assert response.status_code == 200
    assert response.json() == {"species": "virginica"}

def test_zeros_are_rejected():
    #Impossible input: 0cm should give a 422 validation error 
    flower = {"sepal_length": 0, "sepal_width": 0, "petal_length": 0, "petal_width": 0}
    response = client.post("/predict", json=flower)
    assert response.status_code == 422

def test_missing_field_is_rejected():
    #Only 3 of the 4 measurements: should also be rejected 
    flower = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4}
    response = client.post("/predict", json=flower)
    assert response.status_code == 422