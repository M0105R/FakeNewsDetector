import pickle
import joblib
import os

# Path to your model folder
MODEL_DIR = r"C:\Users\monisha\Desktop\monisha\fake_news_detector\model"
PKL_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Show original file size
old_size = os.path.getsize(PKL_PATH) / (1024 * 1024)

# Load and repack with compression
with open(PKL_PATH, "rb") as f:
    vec = pickle.load(f)

joblib.dump(vec, PKL_PATH, compress=3)

# Show new file size
new_size = os.path.getsize(PKL_PATH) / (1024 * 1024)

print(f"✅ Repacked successfully! File: {PKL_PATH}")
print(f"📦 Old size: {old_size:.2f} MB → New size: {new_size:.2f} MB")
print("Now you can safely push it to GitHub 🚀")
