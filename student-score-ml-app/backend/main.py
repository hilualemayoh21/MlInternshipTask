from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Student Score Prediction API")
frontend_url = os.getenv("FRONTEND_URL")
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|[a-z0-9-]+\.vercel\.app)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model/model.pkl")

class StudentInput(BaseModel):
    Hours_Studied: float
    Sleep_Hours: float
    Attendance: float
    Previous_Scores: float

@app.get("/")
def home():
    return {"message": "API Running"}

@app.post("/predict")
def predict(data: StudentInput):
    input_data = np.array([[
        data.Hours_Studied,
        data.Sleep_Hours,
        data.Attendance,
        data.Previous_Scores
    ]])

    prediction = model.predict(input_data)

    return {"Predicted Score": float(prediction[0])}
