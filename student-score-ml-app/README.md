# Student Score ML App

Frontend + FastAPI backend for student exam score prediction.

## Backend
- `cd backend`
- `py -3.10 -m pip install -r requirements.txt`
- `py -3.10 train.py`
- `py -3.10 -m uvicorn main:app --reload`

## Frontend
- `cd frontend`
- `npm install`
- `npm run dev`

Set frontend env in `frontend/.env`:
`VITE_API_URL=http://127.0.0.1:8000`
