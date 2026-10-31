#Small official Python image
FROM python:3.12-slim

#All following commands run inside / app in the container
WORKDIR /app

#Install libraries first (Docker caches this layer, so rebuilds are faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Copy the code
COPY train.py app.py ./

#Train the model during the build, so the image already contains model.joblib
RUN python train.py

#The API listens on port 8000
EXPOSE 8000

# 0.0.0.0 = accept requests from outside the container (127.0.0.1 would block them)
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]