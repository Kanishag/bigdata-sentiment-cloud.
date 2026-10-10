import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import certifi

app = FastAPI(title="Public Sentiment API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

client = MongoClient(os.environ["MONGO_URI"], tlsCAFile=certifi.where())
db = client["sentiment_project"]

@app.get("/")
def home():
    return {"status": "ok"}

@app.get("/overall")
def overall():
    return list(db["overall"].find({}, {"_id": 0}))

@app.get("/topics")
def topics():
    return list(db["topic_sentiment"].find({}, {"_id": 0}))

@app.get("/posts")
def posts(topic: Optional[str] = None, limit: int = 20):
    q = {"topic": topic} if topic else {}
    return list(db["posts"].find(q, {"_id": 0}).limit(limit))
