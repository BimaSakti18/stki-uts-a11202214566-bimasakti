import math
from collections import defaultdict
from typing import Dict, List
from preprocess import preprocess_text

def compute_tf(doc_tokens: List[str]) -> Dict[str,int]:
    tf = defaultdict(int)
    for t in doc_tokens:
        tf[t] += 1
    return dict(tf)

def build_document_term_matrix(docs: Dict[str,str]):
    # docs: doc_id -> raw text
    doc_terms = {doc_id: preprocess_text(text) for doc_id,text in docs.items()}
    tfs = {doc_id: compute_tf(tokens) for doc_id,tokens in doc_terms.items()}
    # DF
    df = defaultdict(int)
    for tf in tfs.values():
        for term in tf.keys():
            df[term] += 1
    N = len(docs)
    idf = {term: math.log(N/df_t) for term, df_t in df.items()}
    # TF-IDF
    tfidf = {}
    for doc_id, tf in tfs.items():
        vec = {term: tf_val * idf[term] for term, tf_val in tf.items()}
        tfidf[doc_id] = vec
    return {
        "doc_terms": doc_terms,
        "tfs": tfs,
        "df": dict(df),
        "idf": idf,
        "tfidf": tfidf
    }

def cosine_similarity(vec_q: Dict[str,float], vec_d: Dict[str,float]) -> float:
    dot = sum(vec_q.get(t,0.0) * vec_d.get(t,0.0) for t in vec_q.keys())
    norm_q = math.sqrt(sum(v*v for v in vec_q.values()))
    norm_d = math.sqrt(sum(v*v for v in vec_d.values()))
    if norm_q == 0 or norm_d == 0:
        return 0.0
    return dot / (norm_q * norm_d)

def rank_query(query: str, model_data, top_k=3):
    q_tokens = preprocess_text(query)
    q_tf = compute_tf(q_tokens)
    q_vec = {t: q_tf[t] * model_data["idf"].get(t, 0.0) for t in q_tf}
    scores = []
    for doc_id, dvec in model_data["tfidf"].items():
        sim = cosine_similarity(q_vec, dvec)
        scores.append((doc_id, sim))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
