# Public Sentiment Analysis Using Big Data, Cloud and AI

An end-to-end big data, cloud and AI system that turns a large collection of social media posts into topic-level insights about public sentiment, and shows them on a live dashboard.

**Live dashboard:** https://bigdata-sentiment-cloud-yz8luddtrzmyzxpqsgpapr.streamlit.app/
**API documentation:** https://sentiment-api-4o4f.onrender.com/docs

## Contents
Abstract, Problem Statement, Objectives, Technologies, Dataset, System Design, Implementation, Results and Analysis, Challenges and Solutions, Security and Privacy, Limitations and Future Work, Conclusion, Project Files, How to Run, References

## Abstract
Social media produces a continuous stream of public opinion, but its size makes it impossible to read manually. This project cleans 300,000 posts from a public Hugging Face dataset with PySpark, labels each post as positive, neutral or negative with VADER, and cross-checks the labels with a RoBERTa transformer model on 2,000 posts (the two agree on 63.65% of posts). Posts are grouped into four topics (work/jobs, school, health, money). Results are stored in MongoDB Atlas, served through a FastAPI REST API on Render, and shown on a Streamlit dashboard.

Overall, 47.2% of posts were positive, 27.6% neutral and 25.2% negative. Health was by far the most negative topic (about 72% negative). A chi-square test (chi-square = 3236.3, 6 degrees of freedom, p < 0.001, Cramer's V = 0.25) shows that sentiment differs significantly between topics. The whole system was built with free tools, entirely from a mobile phone.

## Problem Statement
**Background.** Every day, millions of people share opinions on social media about jobs, health, education and money. These opinions show how the public really feels, but the volume is far too large to read manually.

**Problem.** Governments, companies and institutions still depend on surveys and feedback forms to understand public opinion. Surveys are slow, costly, limited to small samples and often outdated by the time results arrive, so decision-makers react late to public concerns. Raw social media data is huge, unstructured and noisy, so it cannot be analyzed with normal tools.

**Proposed solution.** A cloud-based system that processes a large collection of posts with big data tools (PySpark), classifies each post with AI models, groups the results by topic, stores them in a cloud database, exposes them through a REST API, and shows them on a public dashboard.

**Who benefits.** Policy makers can spot which issues cause the most concern, institutions can track how people feel about services, and researchers get a ready pipeline for large-scale text analysis.

**Scope.** Sentiment by topic on a sample of 300,000 posts. No regional analysis, because the dataset has no location data.

## Objectives
**Main objective:** build a cloud-based system that uses big data and AI to analyze public sentiment from a large collection of social media posts and presents the results on a live dashboard.

1. Process 300,000 social media posts using PySpark.
2. Label every post with VADER and validate the results against a RoBERTa transformer model on a sample.
3. Group posts into four topics and find which are most positive and most negative.
4. Test the findings with a chi-square test.
5. Store the results in MongoDB Atlas.
6. Expose the data through a FastAPI REST API deployed on Render.
7. Visualize the results on an interactive Streamlit dashboard.
8. Document and publish the work on GitHub.

## Technologies
| Service / tool | Used for |
|---|---|
| Google Colab | Running PySpark and the AI models |
| PySpark | Cleaning 300,000 posts |
| VADER | Rule-based sentiment labeling of all posts |
| RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest) | Context-aware validation on a 2,000-post sample |
| MongoDB Atlas | Cloud database (free tier) |
| FastAPI on Render | REST API |
| Streamlit and Plotly on Streamlit Community Cloud | Dashboard |
| GitHub | Code hosting and documentation |

## Dataset
Public Hugging Face dataset `bdstar/Tweets-Sentiment-Analysis`, a merged collection of open tweet sentiment sources.

| Item | Value |
|---|---|
| Total rows (all splits) | 2,526,098 |
| Columns | 6 |
| Rows sampled | 300,000 (random seed 1) |
| Field used | Post text |

**Cleaning (PySpark):** removed duplicate and empty posts, converted text to lowercase, removed links and user mentions, removed symbols and extra spaces.

**Topic assignment:** whole-word keyword matching. In total 26,918 posts matched a topic.

| Topic | Keywords |
|---|---|
| Work/jobs | job, work, boss, unemployed, career, hired, fired |
| School | school, exam, homework, class, teacher, college |
| Health | sick, doctor, hospital, pain, flu, health |
| Money | money, pay, rent, broke, bills, price |

## System Design
```mermaid
flowchart LR
    A[Hugging Face dataset] --> B[PySpark cleaning<br/>Google Colab]
    B --> C[VADER + RoBERTa<br/>sentiment]
    C --> D[(MongoDB Atlas)]
    D --> E[FastAPI<br/>Render]
    E --> F[Streamlit dashboard]
```

**Database** (database `sentiment_project`):

| Collection | Documents | Content |
|---|---|---|
| overall | 3 | Number of posts per sentiment |
| topic_sentiment | 4 | Sentiment counts per topic |
| posts | 5,000 | Random sample of posts with text, sentiment and topic |

**API endpoints:**

| Endpoint | Parameters | Returns |
|---|---|---|
| / | none | Status message |
| /overall | none | Overall sentiment counts |
| /topics | none | Sentiment counts per topic |
| /posts | topic, limit (default 20) | Sample posts, optionally filtered by topic |

**Dashboard:** a pie chart of overall sentiment, a stacked bar chart of sentiment by topic (scaled to 100%), and a table of sample posts that can be filtered by topic. API responses are cached for five minutes.

## Implementation
1. **Setup:** Google Colab with datasets, pyspark, vaderSentiment, transformers and pymongo; free accounts on MongoDB Atlas, Render and Streamlit.
2. **Cleaning:** sampled posts converted to a Spark DataFrame, then cleaned with PySpark functions.
3. **Sentiment:** VADER on every post (compound score of 0.05 or more is positive, -0.05 or less is negative, otherwise neutral). RoBERTa on 2,000 random posts on a GPU (batch size 32, posts truncated to 128 tokens).
4. **Analysis:** topic keywords, percentage tables, stacked bar chart, chi-square test and word counts.
5. **Storage:** summary counts and the 5,000-post sample inserted into MongoDB Atlas with pymongo.
6. **API:** FastAPI app deployed on Render with build command `pip install -r requirements.txt` and start command `uvicorn main:app --host 0.0.0.0 --port $PORT`. The connection string is the `MONGO_URI` environment variable.
7. **Dashboard:** Streamlit app in `dashboard/app.py`, deployed on Streamlit Community Cloud.

## Results and Analysis
**Overall sentiment**

| Sentiment | Share of posts |
|---|---|
| Positive | 47.2% |
| Neutral | 27.6% |
| Negative | 25.2% |

**Sentiment by topic (number of posts)**

| Topic | Posts | Positive | Neutral | Negative |
|---|---|---|---|---|
| Work/jobs | 12,962 | 5,615 | 3,750 | 3,597 |
| School | 6,555 | 2,662 | 2,008 | 1,885 |
| Health | 4,397 | 948 | 302 | 3,147 |
| Money | 3,004 | 1,322 | 398 | 1,284 |

**Sentiment by topic (share of posts)**

| Topic | Positive | Neutral | Negative |
|---|---|---|---|
| Work/jobs | 43.3% | 28.9% | 27.8% |
| School | 40.6% | 30.6% | 28.8% |
| Health | 21.6% | 6.9% | 71.6% |
| Money | 44.0% | 13.2% | 42.7% |

**Interpretation**
- **Health** is the most negative topic by a wide margin. Part of this comes from vocabulary: words such as sick, pain and flu describe unpleasant situations, so the result reflects how people talk about health problems more than opinions about health services.
- **Work/jobs** has the largest volume and is mostly positive.
- **Money** is almost evenly split between positive and negative.
- **School** is the most neutral of the large topics and also leans positive.

**Model comparison:** on 2,000 posts, VADER and RoBERTa gave the same label for 63.65% of posts. They differ most on sarcasm, mixed feelings and neutral text, so VADER percentages are a good overall indication, not exact values.

**Statistical test:** chi-square = 3236.3, 6 degrees of freedom, p < 0.001, Cramer's V = 0.25 (26,918 posts). The differences between topics are statistically significant, and the effect size shows a moderate association between topic and sentiment, mainly driven by health.

**Word frequency:** "good" is the most frequent word in positive posts (14,352 times) and "now" in negative posts (4,908 times).

## Challenges and Solutions
| Challenge | Cause | Solution |
|---|---|---|
| Dataset failed to load in Colab | First dataset used an old loading script that newer library versions do not support | Switched to a dataset in the standard format |
| API crashed on start-up | Connection string typing errors (leftover brackets, curly quotes from the phone keyboard, a stray space, a cut-off host name) | Tested the string in Colab first, then copy-pasted it exactly into Render |
| Authentication failed | Database password in Render did not match Atlas | Reset the database user password and updated the environment variable |
| Slow first response | Free Render server sleeps when unused | 90-second request timeout and cached results |
| Streamlit could not find the repository | Repository was private | Checked that the code contains no secrets, then made it public |
| Building without a laptop | Only a phone was available | Used Colab, the GitHub website and cloud dashboards in a phone browser |

## Security and Privacy
- The database password is stored only as an environment variable on Render, not in the code.
- The dataset is public and has no private user data. Links and user mentions were removed during cleaning.
- For simplicity on free tiers, the database allows connections from any address and the API allows any origin. A production system should restrict both.

## Limitations and Future Work
**Limitations**
- No location field, so analysis is by topic, not by region.
- Keyword topics can misclassify posts, and a post belongs to only one topic.
- VADER is rule-based and can misread sarcasm; only a sample was checked with RoBERTa.
- Only a 300,000-post sample of the 2.5 million rows was used.
- Free hosting may take about 50 seconds to wake up.

**Future work**
- Use a dataset with location data for a regional sentiment map.
- Replace keyword topics with automatic topic modeling.
- Run RoBERTa on all posts and compare with the dataset's own labels.
- Collect posts continuously for near real-time trends.
- Restrict database and API access and add user login.

## Conclusion
A complete big data, cloud and AI pipeline can be built with free tools, even from a mobile phone. It turns a large collection of raw posts into clear, topic-level insights about public sentiment, supported by a statistical test, and makes them available to anyone through a live dashboard. Almost half of the posts are positive, health-related posts are strongly negative, and the differences between topics are statistically significant.

## Project Files
- `sentiment_analysis.ipynb`: data cleaning, sentiment labeling and analysis
- `main.py`: REST API (endpoints /overall, /topics, /posts)
- `requirements.txt`: API dependencies
- `dashboard/app.py`: Streamlit dashboard
- `dashboard/requirements.txt`: dashboard dependencies

## How to Run
1. Run the notebook in Google Colab and set your own MongoDB connection string.
2. Deploy `main.py` on Render with the `MONGO_URI` environment variable.
3. Deploy `dashboard/app.py` on Streamlit Community Cloud.

## References
- Hutto, C. J. and Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM.
- Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
- Loureiro, D. et al. (2022). TimeLMs: Diachronic Language Models from Twitter.
- Zaharia, M. et al. (2016). Apache Spark: A Unified Engine for Big Data Processing. Communications of the ACM.
- Dataset: bdstar/Tweets-Sentiment-Analysis on Hugging Face.
