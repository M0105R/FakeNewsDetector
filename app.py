import streamlit as st
import os, requests, feedparser
from bs4 import BeautifulSoup
import joblib   # use joblib for loading model/vectorizer

# Page config (light theme)
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.markdown(
    """
    <style>
    /* light card style */
    .card {
        background: #ffffff;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .source {
        font-size: 12px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📰 Fake News Detector")
st.write("Empower Yourself with Real Knowledge and Not Misinformation!")

# -------------------------
# Load local ML model
# -------------------------
# ✅ Now files are directly in the project root
BASE_DIR = os.path.dirname(__file__)
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

@st.cache_resource
def load_model():
    """
    Load vectorizer + classifier. Uses joblib for both.
    The vectorizer.pkl was saved using joblib.dump(), so pickle.load() would fail.
    """
    try:
        vec = joblib.load(VEC_PATH)
        clf = joblib.load(MODEL_PATH)
        return vec, clf
    except Exception as e:
        st.error("❌ Local model load failed: " + str(e))
        return None, None

vec, clf = load_model()

# -------------------------
# Fact Check API helper
# -------------------------
def call_google_factcheck(query: str, api_key: str):
    """Query Google Fact Check Tools API and return a list of claim results."""
    if not api_key:
        return []
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"key": api_key, "query": query, "pageSize": 5}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        j = resp.json()
        claims = j.get("claims", [])
        results = []
        for c in claims:
            claim_text = c.get("text", "")
            claimant = c.get("claimant", "")
            reviews = c.get("claimReview", [])
            if reviews:
                review = reviews[0]
                textual_rating = review.get("textualRating", "")
                publisher = review.get("publisher", {}).get("name", "")
                link = review.get("url", "")
                published = review.get("publishedDate", "")
            else:
                textual_rating = publisher = link = published = ""
            results.append({
                "claim": claim_text,
                "claimant": claimant,
                "textual_rating": textual_rating,
                "publisher": publisher,
                "url": link,
                "published": published
            })
        return results
    except Exception as e:
        st.warning("⚠️ Fact Check API error: " + str(e))
        return []

# -------------------------
# RSS sources for auto headlines
# -------------------------
RSS_FEEDS = {
    "Reuters": "http://feeds.reuters.com/reuters/topNews",
    "BBC": "http://feeds.bbci.co.uk/news/rss.xml",
    "CNN": "http://rss.cnn.com/rss/edition.rss",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "NYTimes": "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    "Times of India": "https://timesofindia.indiatimes.com/rssfeedstopstories.cms"
}

# -------------------------
# UI: sidebar controls
# -------------------------
st.sidebar.header("Options")
confidence = st.sidebar.slider("Confidence threshold for REAL", 0.5, 0.99, 0.62, 0.01)
max_feeds = st.sidebar.slider("Max headlines to fetch per source", 1, 10, 5)
use_factcheck = st.sidebar.checkbox("Use Google Fact Check", value=True)
st.sidebar.write("**Default threshold is 0.62 for REAL.**")
st.sidebar.write("**Adjust thresholds as you prefer.**")

# Fetch API key from secrets if present
try:
    gc_key = st.secrets["google"]["fact_check_key"]
except Exception:
    gc_key = None

# -------------------------
# Functions: extract article from URL
# -------------------------
def extract_text_from_url(url: str) -> str:
    """Extract article text using requests + BeautifulSoup."""
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text = "\n\n".join(paragraphs)
        return text if len(text) >= 50 else ""
    except Exception:
        return ""

# -------------------------
# Main: two-column layout
# -------------------------
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("Auto fetch recent headlines")
    if st.button("Fetch latest headlines"):
        st.session_state["fetch_now"] = True
    if st.session_state.get("fetch_now", False):
        headlines, metas = [], []
        for src_name, feed_url in RSS_FEEDS.items():
            try:
                d = feedparser.parse(feed_url)
                for e in d.get("entries", [])[:max_feeds]:
                    title = e.get("title", "")
                    summary = e.get("summary", "") or e.get("description", "")
                    link = e.get("link", "")
                    text = f"{title}. {BeautifulSoup(summary, 'html.parser').get_text()}"
                    headlines.append(text)
                    metas.append({"source": src_name, "title": title, "link": link})
            except Exception as ex:
                st.warning(f"Feed parse failed for {src_name}: {ex}")

        if not headlines:
            st.info("No headlines fetched. Check internet connection.")
        else:
            for i, text in enumerate(headlines):
                meta = metas[i]
                displayed = False
                fact_results = []

                if use_factcheck and gc_key:
                    fact_results = call_google_factcheck(meta["title"], gc_key)

                if fact_results:
                    with st.container():
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"**{meta['title']}** — _{meta['source']}_")
                        st.write(meta["link"])
                        st.write(text[:800])
                        st.markdown("**Fact-check results:**")
                        for r in fact_results:
                            rating = r.get("textual_rating") or "Unknown"
                            publisher = r.get("publisher") or "Unknown"
                            url = r.get("url") or ""
                            published = r.get("published") or ""
                            verdict = rating.lower()
                            if "false" in verdict or "pants on fire" in verdict:
                                st.error(f"{rating} — {publisher} — {published}")
                            elif "true" in verdict or "correct" in verdict or "accurate" in verdict:
                                st.success(f"{rating} — {publisher} — {published}")
                            else:
                                st.info(f"{rating} — {publisher} — {published}")
                            if url:
                                st.write(url)
                        st.markdown("</div>", unsafe_allow_html=True)
                        displayed = True

                if not displayed:
                    if vec is None or clf is None:
                        st.error("Local model not available for fallback classification.")
                    else:
                        X = vec.transform([text])
                        probs = clf.predict_proba(X)[0]
                        classes = list(clf.classes_)
                        prob_dict = {classes[j]: float(probs[j]) for j in range(len(classes))}
                        real_prob = prob_dict.get("REAL", prob_dict.get("Real", 0.0))
                        label = "REAL" if real_prob >= confidence else "FAKE"
                        with st.container():
                            st.markdown("<div class='card'>", unsafe_allow_html=True)
                            st.markdown(f"**{meta['title']}** — _{meta['source']}_")
                            st.write(meta["link"])
                            st.write(text[:800])
                            if label == "REAL":
                                st.success(f"{label} — confidence {real_prob:.2f}")
                            else:
                                st.error(f"{label} — confidence {real_prob:.2f}")
                            if st.button(f"Show raw probs #{i}", key=f"raw_{i}"):
                                st.write(prob_dict)
                            st.markdown("</div>", unsafe_allow_html=True)

with col_side:
    st.subheader("Check a specific article")
    mode = st.radio("Mode", ["Paste text", "Enter URL"])
    user_text = ""
    user_url = ""

    if mode == "Paste text":
        user_text = st.text_area("Paste article text or headline:", height=220)
    else:
        user_url = st.text_input("Enter article URL (http/https):")

    if st.button("Analyze article"):
        if mode == "Enter URL":
            if not user_url:
                st.warning("Enter a URL first.")
            else:
                extracted = extract_text_from_url(user_url)
                if not extracted or len(extracted) < 50:
                    st.warning("Could not extract article text from URL; paste text manually for best results.")
                    user_text = ""
                else:
                    user_text = extracted

        if not user_text or len(user_text.strip()) < 20:
            st.warning("Provide article text (or a valid URL with extractable content).")
        else:
            found = []
            if use_factcheck and gc_key:
                q = user_text[:300]
                found = call_google_factcheck(q, gc_key)

            if found:
                st.markdown("### Fact-check results")
                for fr in found:
                    rating = fr.get("textual_rating") or "Unknown"
                    publisher = fr.get("publisher") or "Unknown"
                    url = fr.get("url") or ""
                    pubd = fr.get("published") or ""
                    if "false" in rating.lower():
                        st.error(f"{rating} — {publisher} — {pubd}")
                    elif "true" in rating.lower():
                        st.success(f"{rating} — {publisher} — {pubd}")
                    else:
                        st.info(f"{rating} — {publisher} — {pubd}")
                    if url:
                        st.write(url)
            else:
                if vec is None or clf is None:
                    st.error("ML fallback not available.")
                else:
                    X = vec.transform([user_text])
                    probs = clf.predict_proba(X)[0]
                    classes = list(clf.classes_)
                    prob_dict = {classes[j]: float(probs[j]) for j in range(len(classes))}
                    real_prob = prob_dict.get("REAL", prob_dict.get("Real", 0.0))
                    label = "REAL" if real_prob >= confidence else "FAKE"
                    st.markdown("###")
                    if label == "REAL":
                        st.success(f"{label} — confidence {real_prob:.2f}")
                    else:
                        st.error(f"{label} — confidence {real_prob:.2f}")
                    st.write("Raw probabilities:")
                    st.write(prob_dict)

st.caption("Fact Check (Google) used.")
