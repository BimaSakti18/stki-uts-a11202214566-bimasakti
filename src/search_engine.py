# src/search_engine.py
import os
import argparse
from math import sqrt, log

# fungsi sederhana untuk preprocessing
def preprocess(text):
    text = text.lower()
    for ch in [",", ".", "!", "?"]:
        text = text.replace(ch, "")
    return text.split()

# bangun model tf-idf
def build_tfidf(docs):
    N = len(docs)
    vocab = {}
    for doc_id, text in docs.items():
        tokens = preprocess(text)
        for t in tokens:
            vocab.setdefault(t, set()).add(doc_id)

    idf = {t: log(N/len(vocab[t])) for t in vocab}
    tfidf = {}
    for doc_id, text in docs.items():
        tokens = preprocess(text)
        tfidf[doc_id] = {}
        for t in tokens:
            tfidf[doc_id][t] = tokens.count(t) * idf[t]
    return tfidf, idf

def cosine(q_vec, d_vec):
    dot = sum(q_vec.get(t, 0)*d_vec.get(t, 0) for t in q_vec)
    nq = sqrt(sum(v*v for v in q_vec.values()))
    nd = sqrt(sum(v*v for v in d_vec.values()))
    return dot / (nq*nd) if nq and nd else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--data", default="data")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    docs = {}
    for fname in os.listdir(args.data):
        if fname.endswith(".txt"):
            with open(os.path.join(args.data, fname), "r", encoding="utf8") as f:
                docs[fname.replace(".txt", "")] = f.read()

    tfidf, idf = build_tfidf(docs)
    q_tokens = preprocess(args.query)
    q_tf = {t: q_tokens.count(t) * idf.get(t, 0) for t in q_tokens}

    scores = [(d, cosine(q_tf, vec)) for d, vec in tfidf.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n=== Hasil Pencarian ===")
    for d, s in scores[:args.k]:
        print(f"{d}\t{s:.4f}\t{docs[d][:60]}")

if __name__ == "__main__":
    main()
