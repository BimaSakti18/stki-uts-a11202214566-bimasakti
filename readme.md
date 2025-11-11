# UTS STKI - Bimasakti (A11.2022.14566)

## Deskripsi
Proyek ini merupakan implementasi sistem **Information Retrieval** sederhana menggunakan:
- **Boolean Model**
- **Vector Space Model (TF-IDF + Cosine Similarity)**

Dilengkapi dengan modul **evaluasi IR** untuk menghitung Precision, Recall, dan MAP.

---
## Cara Menjalankan
1. Aktifkan virtual environment: ..venv\Scripts\activate
2. Jalankan sistem pencarian: 'python src/search_engine.py --query "kucing ikan" --k 3 --data data/'
3. Jalankan evaluasi: 'python src/eval.py'
4. Jalankan notebook di Jupyter 

## Hasil Singkat
| Query | Precision@3 | Recall@3 | AP |
|-------|--------------|----------|----|
| kucing ikan | 0.667 | 1.0 | 1.0 |
| anjing taman | 0.667 | 1.0 | 1.0 |
| **MAP** |  |  | **1.0** |

---

## 👨‍💻 Pembuat
**Nama:** Bimasakti  
**NIM:** A11.2022.14566  
**Kelas:** A11.4703  
**Mata Kuliah:** Sistem Temu Kembali Informasi