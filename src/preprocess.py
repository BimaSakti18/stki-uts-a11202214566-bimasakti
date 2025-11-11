import re
from typing import List

# contoh daftar stopword Bahasa Indonesia (ringkas)
STOPWORDS = {
    "di","dan","yang","ke","dari","ke","untuk","oleh","pada","adalah","ini","itu","sebagai","dgn","dg"
}

def clean(text: str) -> str:
    text = text.lower()
    # normalisasi angka
    text = re.sub(r'\d+([.,]\d+)?', ' <NUM> ', text)
    # hapus tanda baca
    text = re.sub(r'[^\w\s]', ' ', text)
    # pad trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return text.split()

def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]

# stemming ringan: hapus akhiran umum
def stem(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        #rules sederhana
        t0 = re.sub(r'(lah|kah|ku|mu|nya)$', '', t)
        t0 = re.sub(r'(kan|i|an)$', '', t0)
        out.append(t0)
    return out

def preprocess_text(text: str) -> List[str]:
    c = clean(text)
    toks = tokenize(c)
    toks = remove_stopwords(toks)
    toks = stem(toks)
    return toks

if __name__ == "__main__":
    sample = "Kucing makan ikan dan bermain di halaman."
    print("before:", sample)
    print("after:", preprocess_text(sample))
