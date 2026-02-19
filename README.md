# IDFC FIRST Bank — AI Underwriting Copilot v2.0

## Start in 3 steps

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your API keys
uvicorn app.main:app --reload --port 8000
```

## Seed Weaviate (once)
```bash
curl -X POST http://localhost:8000/policy/seed
```

## Open frontend
Open `frontend/index.html` with VS Code Live Server, or:
```bash
python -m http.server 3000 --directory frontend
```
Visit http://localhost:3000

## API Docs
http://localhost:8000/api/docs

## Health Check
http://localhost:8000/health
