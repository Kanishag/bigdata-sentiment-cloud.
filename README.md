# bigdata-sentiment-cloud.# Public Sentiment Analysis using Big Data, Cloud and AI

## Problem statement
Governments and organizations find it hard to understand public opinion quickly, because surveys are slow and costly. This project analyzes large volumes of social media posts with AI, classifies their sentiment by topic, and shows the results on a cloud-hosted dashboard so decision-makers can see public mood almost immediately.

## Live demo
- Dashboard: https://bigdata-sentiment-cloud-yz8luddtrzmyzxpqsgpapr.streamlit.app/
- API: https://sentiment-api-4o4f.onrender.com/docs

## Architecture
Dataset (Hugging Face) -> PySpark cleaning (Google Colab) -> VADER and RoBERTa sentiment models -> MongoDB Atlas (cloud database) -> FastAPI on Render -> Streamlit dashboard

## Tech stack
- Big data: PySpark, pandas
- AI: VADER, cardiffnlp/twitter-roberta-base-sentiment-latest
- Cloud: MongoDB Atlas, Render, Streamlit Community Cloud
- Backend: FastAPI

## Key results
- Overall sentiment: 47.2% positive, 27.6% neutral, 25.2% negative
- Health posts were the most negative (about 72% negative)
- Work/jobs posts had the most volume and were mostly positive

## Project files
- sentiment_analysis.ipynb: data cleaning, sentiment labeling and analysis
- main.py: REST API (endpoints /overall, /topics, /posts)
- dashboard/app.py: Streamlit dashboard

## Limitations
- The dataset has no location data, so analysis is by topic, not region
- Topics are assigned by keyword matching
- Free-tier hosting may take about 50 seconds to wake up

## How to run
1. Run the notebook in Google Colab and set your own MongoDB connection string.
2. Deploy main.py on Render with the MONGO_URI environment variable.
3. Deploy dashboard/app.py on Streamlit Cloud.
