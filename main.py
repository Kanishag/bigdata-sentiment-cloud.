import os
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import certifi

app = FastAPI(title="Public Sentiment API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

client = MongoClient(os.environ["MONGO_URI"], tlsCAFile=certifi.where())
db = client["sentiment_project"]
vader = SentimentIntensityAnalyzer()

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
def posts(topic: Optional[str] = None, sentiment: Optional[str] = None, limit: int = 20):
    q = {}
    if topic:
        q["topic"] = topic
    if sentiment:
        q["vader"] = sentiment
    return list(db["posts"].find(q, {"_id": 0}).limit(min(limit, 5000)))

@app.get("/predict")
def predict(text: str):
    scores = vader.polarity_scores(text)
    c = scores["compound"]
    label = "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"
    return {"label": label, "compound": c, "scores": scores}
