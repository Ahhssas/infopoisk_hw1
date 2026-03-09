from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def build_vocab(tokenized_docs):
    feature_names = sorted(set(term for doc in tokenized_docs for term in doc))
    vocabulary = {term: idx for idx, term in enumerate(feature_names)}
    return feature_names, vocabulary


def build_tf_matrix(tokenized_docs):
    feature_names, vocabulary = build_vocab(tokenized_docs)

    rows, cols, data = [], [], []
    for doc_id, tokens in enumerate(tokenized_docs):
        counts = Counter(tokens)
        for term, tf in counts.items():
            rows.append(doc_id)
            cols.append(vocabulary[term])
            data.append(tf)

    tf_matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(tokenized_docs), len(feature_names)),
        dtype=float,
    )
    return tf_matrix, feature_names, vocabulary


def build_bm25_matrix(tf_matrix, k1=1.5, b=0.75):
    n_docs = tf_matrix.shape[0]
    doc_lengths = np.asarray(tf_matrix.sum(axis=1)).ravel()
    avgdl = doc_lengths.mean() if n_docs > 0 else 0.0

    df = np.asarray((tf_matrix > 0).sum(axis=0)).ravel()
    idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)

    bm25_coo = tf_matrix.tocoo(copy=True)

    for i in range(len(bm25_coo.data)):
        doc_id = bm25_coo.row[i]
        term_id = bm25_coo.col[i]
        tf = bm25_coo.data[i]
        dl = doc_lengths[doc_id]

        bm25_coo.data[i] = idf[term_id] * (
            tf * (k1 + 1) /
            (tf + k1 * (1 - b + b * dl / avgdl))
        )

    bm25_matrix = bm25_coo.tocsr()

    stats = {
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
    }
    return bm25_matrix, stats


def make_query_vector(query, vocabulary, preprocess_func=None, binary=False):
    if preprocess_func:
        query = preprocess_func(query)

    q = np.zeros(len(vocabulary), dtype=float)
    terms = query.split()

    if binary:
        for term in set(terms):
            if term in vocabulary:
                q[vocabulary[term]] = 1.0
    else:
        counts = Counter(terms)
        for term, tf in counts.items():
            if term in vocabulary:
                q[vocabulary[term]] = tf

    return q


def search_freq_matrix(query, tf_matrix, vocabulary, preprocess_func=None, top_k=10):
    q = make_query_vector(query, vocabulary, preprocess_func=preprocess_func, binary=False)
    scores = np.asarray(tf_matrix @ q).ravel()

    top_idx = np.argsort(scores)[::-1]
    top_idx = top_idx[scores[top_idx] > 0][:top_k]

    return pd.Series(scores[top_idx], index=top_idx).sort_values(ascending=False)


def search_bm25_matrix(query, bm25_matrix, vocabulary, preprocess_func=None, top_k=10):
    q = make_query_vector(query, vocabulary, preprocess_func=preprocess_func, binary=True)
    scores = np.asarray(bm25_matrix @ q).ravel()

    top_idx = np.argsort(scores)[::-1]
    top_idx = top_idx[scores[top_idx] > 0][:top_k]

    return pd.Series(scores[top_idx], index=top_idx).sort_values(ascending=False)
