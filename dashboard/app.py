import requests
import pandas as pd
import plotly.express as px
import streamlit as st

API = "https://sentiment-api-4o4f.onrender.com"

st.set_page_config(page_title="Public Sentiment Dashboard", layout="wide")
st.title("Public Sentiment Analysis")
st.caption("Big data + cloud + AI: sentiment of social media posts by topic")

@st.cache_data(ttl=300)
def get(path):
    return requests.get(API + path, timeout=90).json()

overall = pd.DataFrame(get("/overall"))
topics = pd.DataFrame(get("/topics"))

c1, c2 = st.columns(2)
with c1:
    st.subheader("Overall sentiment")
    st.plotly_chart(px.pie(overall, names="sentiment", values="count"),
                    use_container_width=True)
with c2:
    st.subheader("Sentiment by topic")
    long = topics.melt(id_vars="topic", var_name="sentiment", value_name="count")
    fig = px.bar(long, x="topic", y="count", color="sentiment", barmode="stack")
    fig.update_layout(barnorm="percent")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Explore sample posts")
choice = st.selectbox("Topic", topics["topic"].tolist())
posts = pd.DataFrame(get(f"/posts?topic={choice}&limit=20"))
st.dataframe(posts[["text", "vader"]], use_container_width=True)
