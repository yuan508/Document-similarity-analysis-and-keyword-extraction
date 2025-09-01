# -*- coding: utf-8 -*-
"""
Document similarity demo
Supports:
 - small internal sample dataset (fast, no download)
 - 20 Newsgroups dataset (sklearn fetch)
Methods:
 - TF-IDF (sklearn) + cosine
 - Gensim TF-IDF + SparseMatrixSimilarity
 - SimHash distance (hamming on simhash)
"""
import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.datasets import fetch_20newsgroups
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Optional imports (gensim, simhash). If not installed, those methods will be disabled.
try:
    from gensim import corpora, models, similarities
    _HAS_GENSIM = True
except Exception:
    _HAS_GENSIM = False

try:
    from simhash import Simhash
    _HAS_SIMHASH = True
except Exception:
    _HAS_SIMHASH = False

# -------------------------
# small sample dataset (works offline)
# -------------------------
SAMPLE_DOCS = [
    "Apple releases new iPhone with improved battery life and camera.",
    "Local bakery introduces a new sourdough recipe that sells out daily.",
    "Researchers publish a paper about lithium-ion battery degradation mechanisms.",
    "Stock markets rally on positive jobs report and easing inflation concerns.",
    "New study shows coffee consumption may reduce risk of certain diseases.",
    "Electric vehicles adoption is rising as battery costs drop.",
    "The football team won the championship after a dramatic penalty shootout.",
    "A novel approach for domain adaptation in battery state estimation is proposed.",
    "The restaurant's seasonal menu features locally sourced vegetables and herbs.",
    "Scientists develop faster charging methods for lithium batteries."
]

# -------------------------
# utilities
# -------------------------
def simple_preprocess(text, lowercase=True, remove_punct=True):
    if lowercase:
        text = text.lower()
    if remove_punct:
        # keep simple tokenization: replace non-word chars with space
        text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_corpus(docs, lowercase=True, remove_punct=True):
    return [simple_preprocess(d, lowercase, remove_punct) for d in docs]

# -------------------------
# TF-IDF (sklearn) + cosine
# -------------------------
def build_tfidf_index_sklearn(docs, max_features=20000, ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(docs)  # shape (n_docs, n_features)
    return vectorizer, tfidf_matrix

def query_tfidf_sklearn(vectorizer, tfidf_matrix, query, topk=5):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, tfidf_matrix)[0]
    top_idx = np.argsort(-sims)[:topk]
    return list(zip(top_idx, sims[top_idx]))

# -------------------------
# Gensim TF-IDF + SparseMatrixSimilarity
# -------------------------
def build_gensim_index(docs):
    if not _HAS_GENSIM:
        raise RuntimeError("gensim is required for build_gensim_index")
    texts = [d.split() for d in docs]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    return dictionary, tfidf, index, corpus

def query_gensim(dictionary, tfidf, index, query, topk=5):
    if not _HAS_GENSIM:
        raise RuntimeError("gensim is required for query_gensim")
    qtokens = query.split()
    qbow = dictionary.doc2bow(qtokens)
    qtfidf = tfidf[qbow]
    sims = index[qtfidf]  # numpy array of similarities
    top_idx = np.argsort(-sims)[:topk]
    return list(zip(top_idx, sims[top_idx]))

# -------------------------
# SimHash-based approximate similarity (hamming distance)
# -------------------------
def build_simhashs(docs):
    if not _HAS_SIMHASH:
        raise RuntimeError("simhash is required for build_simhashs")
    return [Simhash(d.split()) for d in docs]

def query_simhash(simhash_list, query, topk=5):
    if not _HAS_SIMHASH:
        raise RuntimeError("simhash is required for query_simhash")
    qh = Simhash(query.split())
    distances = [qh.distance(h) for h in simhash_list]
    # smaller distance => more similar (0 = exact same hash)
    top_idx = np.argsort(distances)[:topk]
    return list(zip(top_idx, np.array(distances)[top_idx]))

# -------------------------
# demo runner
# -------------------------
def demo(use_20newsgroups=True, sample_mode=False):
    if use_20newsgroups:
        print("Loading 20 Newsgroups (may download ~14MB)...")
        data = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
        docs = list(data.data[:2000])  # limit to first 2000 for speed
        titles = [f"doc_{i}" for i in range(len(docs))]
    elif sample_mode:
        docs = SAMPLE_DOCS
        titles = [f"sample_{i}" for i in range(len(docs))]
    else:
        docs = SAMPLE_DOCS
        titles = [f"doc_{i}" for i in range(len(docs))]

    print(f"Total docs: {len(docs)}")
    docs_proc = preprocess_corpus(docs)

    # pick a query doc (an index)
    q_idx = 2  # changeable
    query = docs_proc[q_idx]
    print("\n=== Query document (index {}) ===".format(q_idx))
    print(docs[q_idx][:400])
    print("preprocessed:", query)

    # 1) sklearn TF-IDF
    print("\n--- sklearn TF-IDF + cosine ---")
    vec, tfidf_mat = build_tfidf_index_sklearn(docs_proc)
    top = query_tfidf_sklearn(vec, tfidf_mat, query)
    for idx, score in top:
        print(f"idx={idx:4d} score={score:.4f} title={titles[idx]} preview={docs[idx][:80]!r}")

    # 2) gensim TF-IDF (if available)
    if _HAS_GENSIM:
        print("\n--- gensim TF-IDF + SparseMatrixSimilarity ---")
        dictionary, tfidf, index, corpus = build_gensim_index(docs_proc)
        topg = query_gensim(dictionary, tfidf, index, query)
        for idx, score in topg:
            print(f"idx={idx:4d} score={score:.4f} title={titles[idx]} preview={docs[idx][:80]!r}")
    else:
        print("\n(gensim not installed — skip gensim variant)")

    # 3) SimHash (if available)
    if _HAS_SIMHASH:
        print("\n--- SimHash (Hamming distance) ---")
        sh_list = build_simhashs(docs_proc)
        topsh = query_simhash(sh_list, query)
        for idx, dist in topsh:
            print(f"idx={idx:4d} hamming_dist={dist:3d} title={titles[idx]} preview={docs[idx][:80]!r}")
    else:
        print("\n(simhash not installed — skip simhash variant)")

if __name__ == "__main__":
    #demo(use_20newsgroups=False, sample_mode=True)
    demo(use_20newsgroups=True, sample_mode=False)
