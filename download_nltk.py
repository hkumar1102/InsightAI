import nltk

print("Downloading NLTK dependencies...")
nltk.download('punkt')
try:
    # punkt_tab exists in newer NLTK versions; older versions may not provide it.
    nltk.download('punkt_tab')
except Exception:
    pass
print("Done! Dependencies ready.")
