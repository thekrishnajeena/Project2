from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from solver.runner import solve_quiz_task

load_dotenv()

app = FastAPI()

QUIZ_SECRET = os.getenv("QUIZ_SECRET")


class QuizRequest(BaseModel):
    email: str
    secret: str
    url: str

@app.get("/")
async def project2():
    return {"message": "Everything's fine"}

@app.post("/api/quiz")
async def receive_quiz(payload: QuizRequest, background_tasks: BackgroundTasks):
    if payload.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    background_tasks.add_task(
        solve_quiz_task,
        payload.email,
        payload.secret,
        payload.url
    )

    return {"received": True}
