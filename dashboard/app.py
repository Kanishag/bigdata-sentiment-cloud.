import math
import re
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API = "https://sentiment-api-4o4f.onrender.com"
REPO = "https://github.com/Kanishag/bigdata-sentiment-cloud."
ORDER = ["positive", "neutral", "negative"]
COLORS = {"positive": "#2ECC71", "neutral": "#F4B400", "negative": "#E74C3C"}
STOP = set(
    "the and for you that this with have was are but not just your all out get can now like they what "
    "when will one how its about from has had she him her his them our their there been would could than "
    "then too very really also into over only some more day today going got did does his who why where "
    "which while were being here still even off any let say see way want make much well back".split()
    + ["im", "dont", "cant", "ive", "thats", "amp", "its", "didnt", "isnt", "wont", "youre"]
)


# ---------------------------------------------------------------- helpers
def words_of(text):
    words = re.findall(r"[a-z']{3,}", str(text).lower())
    return [w for w in words if w not in STOP and not w.startswith("'")]


def word_stats(df, top_n=45):
    counts = Counter()
    by_sent = {}
    for text, sent in zip(df["text"], df["vader"]):
        for w in words_of(text):
            counts[w] += 1
            by_sent.setdefault(w, Counter())[sent] += 1
    rows = []
    for w, c in counts.most_common(top_n):
        rows.append((w, c, by_sent[w].most_common(1)[0][0]))
    return rows


def cloud_layout(rows, min_size=13, max_size=32):
    """Place words on a spiral so that their boxes never overlap."""
    mx, mn = rows[0][1], rows[-1][1]
    placed, out = [], []
    for w, c, dom in rows:
        size = min_size + (max_size - min_size) * (c - mn) / (mx - mn + 1e-9)
        bw, bh = len(w) * size * 0.60 + 8, size * 1.3
        t = 0.0
        while True:
            x, y = 3.2 * t * math.cos(t), 2.4 * t * math.sin(t)
            box = (x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2)
            clash = any(not (box[2] < p[0] or box[0] > p[2] or box[3] < p[1] or box[1] > p[3]) for p in placed)
            if not clash or t > 400:
                break
            t += 0.08
        placed.append(box)
        out.append((w, c, dom, x, y, size))
    return out, placed


def cloud_figure(rows):
    items, boxes = cloud_layout(rows)
    fig = go.Figure(
        go.Scatter(
            x=[i[3] for i in items],
            y=[i[4] for i in items],
            mode="text",
            text=[i[0] for i in items],
            textfont=dict(size=[i[5] for i in items], color=[COLORS.get(i[2], "#888888") for i in items]),
            hovertext=[f"{i[0]}: {i[1]} posts" for i in items],
            hoverinfo="text",
        )
    )
    x0, x1 = min(b[0] for b in boxes) - 10, max(b[2] for b in boxes) + 10
    y0, y1 = min(b[1] for b in boxes) - 10, max(b[3] for b in boxes) + 10
    fig.update_layout(
        xaxis=dict(visible=False, range=[x0, x1]),
        yaxis=dict(visible=False, range=[y0, y1]),
        height=480,
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(ttl=300, show_spinner="Waking up the cloud API (the first load can take up to a minute)...")
def get(path, limit=0):
    params = {"limit": limit} if limit else None
    r = requests.get(API + path, params=params, timeout=90)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Public Sentiment Dashboard", page_icon="📊", layout="wide")

st.markdown(
    """
<style>
.hero {background: linear-gradient(120deg, #1F3864, #7B1FA2 55%, #E25A1C); padding: 26px 30px;
       border-radius: 18px; color: white;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: 6px 0 0; opacity: 0.92; font-size: 1.02rem;}
div[data-testid="stMetric"] {background: rgba(127,127,127,0.10); border: 1px solid rgba(127,127,127,0.25);
       padding: 14px 18px; border-radius: 14px;}
.card {background: rgba(127,127,127,0.10); border: 1px solid rgba(127,127,127,0.25);
       padding: 14px 18px; border-radius: 14px; margin-bottom: 12px;}
.card b {font-size: 1.05rem;}
.badge {display: inline-block; padding: 8px 20px; border-radius: 999px; color: white;
        font-weight: 700; font-size: 1.1rem; letter-spacing: 1px;}
</style>
<div class="hero">
  <h1>Public Sentiment Dashboard</h1>
  <p>Big Data + Cloud + AI: how people feel about work, school, health and money, based on 300,000 social media posts.</p>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

try:
    overall = pd.DataFrame(get("/overall"))
    topics = pd.DataFrame(get("/topics"))
    posts = pd.DataFrame(get("/posts", 5000))
except Exception:
    st.error("The cloud API is waking up or not reachable. Please reload this page in a minute.")
    st.stop()

# sidebar filters
st.sidebar.header("Filters")
all_topics = sorted(topics["topic"].tolist())
sel_topics = st.sidebar.multiselect("Topics", all_topics, default=all_topics)
sel_sent = st.sidebar.multiselect("Sentiment", ORDER, default=ORDER)
st.sidebar.caption("Topic filter: Topics, Word cloud and Explore tabs. Sentiment filter: Word cloud and Explore tabs.")

ftopics = topics[topics["topic"].isin(sel_topics)]
fposts = posts[posts["topic"].isin(sel_topics) & posts["vader"].isin(sel_sent)]

# KPI cards
cnt = dict(zip(overall["sentiment"], overall["count"]))
tot = max(sum(cnt.values()), 1)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Posts analysed", f"{tot:,}")
k2.metric("Positive", f"{100 * cnt.get('positive', 0) / tot:.1f}%")
k3.metric("Neutral", f"{100 * cnt.get('neutral', 0) / tot:.1f}%")
k4.metric("Negative", f"{100 * cnt.get('negative', 0) / tot:.1f}%")
st.write("")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Topics", "Word cloud", "Try it yourself", "Explore posts", "About"]
)

# ---- Overview
with tab1:
    left, right = st.columns([3, 2])
    with left:
        fig = px.pie(overall, names="sentiment", values="count", hole=0.55,
                     color="sentiment", color_discrete_map=COLORS)
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=380)
        st.plotly_chart(fig)
    with right:
        long = topics.melt(id_vars="topic", var_name="sentiment", value_name="count")
        long["share"] = 100 * long["count"] / long.groupby("topic")["count"].transform("sum")
        neg = long[long["sentiment"] == "negative"].sort_values("share", ascending=False).iloc[0]
        pos = long[long["sentiment"] == "positive"].sort_values("share", ascending=False).iloc[0]
        tv = topics.assign(total=topics[["negative", "neutral", "positive"]].sum(axis=1)).sort_values("total", ascending=False).iloc[0]
        st.markdown(f'<div class="card">Most negative topic<br><b>{neg["topic"]}</b> ({neg["share"]:.1f}% negative)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">Most positive topic<br><b>{pos["topic"]}</b> ({pos["share"]:.1f}% positive)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card">Most discussed topic<br><b>{tv["topic"]}</b> ({int(tv["total"]):,} posts)</div>', unsafe_allow_html=True)

# ---- Topics
with tab2:
    if ftopics.empty:
        st.info("Select at least one topic in the sidebar.")
    else:
        t_long = ftopics.melt(id_vars="topic", var_name="sentiment", value_name="count")
        t_long["share"] = 100 * t_long["count"] / t_long.groupby("topic")["count"].transform("sum")
        fig = px.bar(t_long, x="topic", y="share", color="sentiment", color_discrete_map=COLORS,
                     category_orders={"sentiment": ORDER}, text=t_long["share"].round(1).astype(str) + "%")
        fig.update_layout(yaxis_title="% of posts", xaxis_title="", yaxis_range=[0, 100],
                          legend_title="", margin=dict(l=0, r=0, t=10, b=0), height=420)
        st.plotly_chart(fig)
        tbl = ftopics.copy()
        tbl["posts"] = tbl[["negative", "neutral", "positive"]].sum(axis=1)
        for s in ORDER:
            tbl[s + " %"] = (100 * tbl[s] / tbl["posts"]).round(1)
        st.dataframe(tbl[["topic", "posts"] + [s + " %" for s in ORDER]].sort_values("posts", ascending=False),
                     hide_index=True)

# ---- Word cloud
with tab3:
    st.caption("Most frequent words in a sample of 5,000 posts. Green words appear mostly in positive posts, "
               "amber in neutral posts and red in negative posts. Bigger means more frequent.")
    if fposts.empty:
        st.info("No posts match the current filters.")
    else:
        wrows = word_stats(fposts)
        if wrows:
            st.plotly_chart(cloud_figure(wrows))
        else:
            st.info("Not enough words to draw a word cloud.")

# ---- Try it yourself
with tab4:
    st.write("Type any sentence and the cloud API will tell you its sentiment, using the VADER model.")
    txt = st.text_area("Your text", "I love how fast this dashboard loads!", height=100)
    if st.button("Analyze sentiment", type="primary"):
        if not txt.strip():
            st.warning("Please type some text first.")
        else:
            try:
                with st.spinner("Analyzing..."):
                    res = requests.get(API + "/predict", params={"text": txt}, timeout=90).json()
                label, comp, sc = res["label"], res["compound"], res["scores"]
                st.markdown(f'<span class="badge" style="background:{COLORS[label]}">{label.upper()}</span>',
                            unsafe_allow_html=True)
                g1, g2 = st.columns(2)
                with g1:
                    gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=comp, title={"text": "Compound score"},
                        gauge={"axis": {"range": [-1, 1]}, "bar": {"color": COLORS[label]},
                               "steps": [{"range": [-1, -0.05], "color": "#FADBD8"},
                                         {"range": [-0.05, 0.05], "color": "#FCF3CF"},
                                         {"range": [0.05, 1], "color": "#D5F5E3"}]}))
                    gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
                    st.plotly_chart(gauge)
                with g2:
                    parts = pd.DataFrame({"sentiment": ORDER, "share": [sc["pos"], sc["neu"], sc["neg"]]})
                    pf = px.bar(parts, x="share", y="sentiment", orientation="h", color="sentiment",
                                color_discrete_map=COLORS, range_x=[0, 1])
                    pf.update_layout(showlegend=False, height=280, xaxis_title="Share of the text", yaxis_title="",
                                     margin=dict(l=10, r=10, t=50, b=10), title="Sentiment mix")
                    st.plotly_chart(pf)
            except Exception:
                st.error("Could not reach the API. It may be waking up, so please try again in a minute.")

# ---- Explore posts
with tab5:
    kw = st.text_input("Search inside the posts (optional)", "")
    view = fposts
    if kw.strip():
        view = view[view["text"].str.contains(kw.strip(), case=False, na=False, regex=False)]
    st.caption(f"{len(view):,} posts match (the dashboard holds a random sample of 5,000 posts).")
    st.dataframe(view[["text", "vader", "topic"]].rename(columns={"vader": "sentiment"}).head(300),
                 hide_index=True)

# ---- About
with tab6:
    st.markdown(
        f"""
**Public Sentiment Analysis Using Big Data, Cloud and AI**

This dashboard shows how people feel about work, school, health and money. 300,000 social media posts were
cleaned with **PySpark**, labeled with **VADER** (checked against a **RoBERTa** model on 2,000 posts),
stored in **MongoDB Atlas**, and served by a **FastAPI** API on **Render**.

- Health is the most negative topic, and the differences between topics are statistically significant
  (chi-square test, p < 0.001).
- The "Try it yourself" tab calls the live API.
- Source code and report: [GitHub repository]({REPO})
        """
    )

st.caption("Built with PySpark, VADER, RoBERTa, MongoDB Atlas, FastAPI, Streamlit and Plotly.")
