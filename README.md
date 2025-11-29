---
title: My FastAPI Space
sdk: docker
app_port: 7860
---

<<<<<<< HEAD
Starter scaffold for the data-quiz solver project using Python, FastAPI, and Playwright.


## Quick start (local)
1. Clone the repo.
2. Create a virtual environment and activate it:


```bash
python -m venv .venv
source .venv/bin/activate # mac/linux
.venv\Scripts\activate # windows
```


3. Install dependencies:


```bash
pip install -r requirements.txt
```


4. Install Playwright browsers (required):


```bash
playwright install
```


5. Copy `.env.example` to `.env` and set `QUIZ_SECRET` and `API_KEY` for your LLM provider.


6. Run the app:


```bash
uvicorn app.main:app --reload
```


7. Test the endpoint (example):


```bash
curl -X POST https://localhost:8000/api/quiz -H "Content-Type: application/json" -d '{"email":"you@example.com","secret":"your-secret","url":"https://tds-llm-analysis.s-anand.net/demo"}'
```
=======
---
title: Project2 Final
emoji: 📚
colorFrom: purple
colorTo: red
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> huggingface/main
