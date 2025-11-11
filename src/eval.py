# src/eval.py
"""
Modul evaluasi Information Retrieval
-------------------------------------
Metrik yang dihitung:
- Precision@k
- Recall@k
- Average Precision (AP)
- Mean Average Precision (MAP)
"""

from typing import List, Dict, Set

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Hitung precision pada posisi k"""
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    hit = len(set(retrieved_k) & relevant)
    return hit / k

def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Hitung recall pada posisi k"""
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hit = len(set(retrieved_k) & relevant)
    return hit / len(relevant)

def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Hitung Average Precision (AP) untuk satu query"""
    if not relevant:
        return 0.0
    precisions = []
    hit = 0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hit += 1
            precisions.append(hit / i)
    if not precisions:
        return 0.0
    return sum(precisions) / len(relevant)

def mean_average_precision(all_retrieved: Dict[str, List[str]], all_relevant: Dict[str, Set[str]]) -> float:
    """Hitung MAP untuk banyak query"""
    ap_sum = 0
    n_queries = len(all_relevant)
    for qid, relevant in all_relevant.items():
        retrieved = all_retrieved.get(qid, [])
        ap_sum += average_precision(retrieved, relevant)
    return ap_sum / n_queries if n_queries else 0.0

def evaluate_queries(all_retrieved: Dict[str, List[str]], all_relevant: Dict[str, Set[str]], k: int = 3):
    """Evaluasi lengkap per query"""
    print(f"\n=== HASIL EVALUASI (k={k}) ===")
    for qid, relevant in all_relevant.items():
        retrieved = all_retrieved.get(qid, [])
        p = precision_at_k(retrieved, relevant, k)
        r = recall_at_k(retrieved, relevant, k)
        ap = average_precision(retrieved, relevant)
        print(f"Query '{qid}': Precision@{k}={p:.3f}, Recall@{k}={r:.3f}, AP={ap:.3f}")
    map_score = mean_average_precision(all_retrieved, all_relevant)
    print(f"\nMean Average Precision (MAP): {map_score:.3f}")

# Contoh penggunaan
if __name__ == "__main__":
    # Contoh hasil pencarian
    retrieved_docs = {
        "kucing ikan": ["doc1", "doc3", "doc2"],   # urutan hasil retrieval
        "anjing taman": ["doc2", "doc4", "doc5"]
    }

    # Ground truth relevansi (harus kamu isi sesuai data uji kamu)
    relevant_docs = {
        "kucing ikan": {"doc1", "doc3"},
        "anjing taman": {"doc2", "doc4"}
    }

    evaluate_queries(retrieved_docs, relevant_docs, k=3)
