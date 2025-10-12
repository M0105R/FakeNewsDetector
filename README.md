# FAKE NEWS DETECTOR

## What this web app does
- Tries Google Fact Check Tools API for a given article.
- If fact-check entries are found, displays verified verdicts and sources.
- If no fact-check result exists, uses a Kaggle-trained TF-IDF + Logistic Regression model as fallback.
- Auto-fetches recent headlines via RSS and classifies them.

## How to run
1. Install dependencies:
   pip install -r requirements.txt

3. (Optional) Add Google Fact Check API key to `.streamlit/secrets.toml`:
   [google]
   fact_check_key = "YOUR_KEY"

4. Run the app:
   streamlit run app.py

Notes:
- If you don't provide a Fact Check API key, the app will still function using the ML fallback.
- For best extraction from URLs, ensure `newspaper3k` is installed (`pip install newspaper3k`) and NLTK punkt available.
