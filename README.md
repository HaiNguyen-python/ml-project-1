# Iris Flower Classifier: from model to Docker API

A small end-to-end machine learning project: train a model, serve it as a
web API, validate the input, and run everything in a Docker container.

## What it does
- **Train** (`train.py`): loads the Iris dataset (150 flowers, 4 measurements),
  compares Logistic Regression and a Decision Tree with 5-fold cross-validation,
  trains a Decision Tree on 80% of the data, tests it on the other 20%,
  and saves it as `model.joblib`.
- **Serve** (`app.py`): a FastAPI web API that loads the model and predicts
  the species (setosa, versicolor, virginica).
- **Validate**: every measurement must be greater than 0 and at most 10 cm;
  impossible inputs (e.g. all zeros) get a `422` error instead of a fake prediction.
- **Containerize** (`Dockerfile`, `compose.yaml`): the image installs the
  libraries and trains the model during the build; Compose runs the API with
  automatic restart and a health check.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check: `{"status": "ok"}` |
| POST | `/predict` | Predict the species from 4 measurements |
| GET | `/docs` | Interactive API docs (Swagger UI) |

Example request:
```json
{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
```
Response: `{"species": "setosa"}`

## How to run

### With Docker (recommended)
```
docker compose up --build -d   # build and start in the background
docker compose ps              # status (becomes "healthy")
docker compose down            # stop
```
Then open http://localhost:8000/docs

### Without Docker
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
python app.py
```

## Tech
Python, scikit-learn, joblib, FastAPI, Pydantic, Uvicorn, Docker, Docker Compose
