from collections import defaultdict, Counter
import math
import pandas as pd


def prepare_tokenized_docs(df, clean_col="text_clean"):
    docs = df[clean_col].fillna("").tolist()
    tokenized_docs = [doc.split() for doc in docs]
    return docs, tokenized_docs


def build_freq_inverted_index(tokenized_docs):
    inverted_index = defaultdict(dict)

    for doc_id, tokens in enumerate(tokenized_docs):
        term_counts = Counter(tokens)
        for term, tf in term_counts.items():
            inverted_index[term][doc_id] = tf

    return dict(inverted_index)


def build_bm25_stats(index, tokenized_docs):
    N = len(tokenized_docs)
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avgdl = sum(doc_lengths) / N if N > 0 else 0.0

    df_terms = {term: len(postings) for term, postings in index.items()}
    idf = {
        term: math.log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df in df_terms.items()
    }

    return {
        "N": N,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "df_terms": df_terms,
        "idf": idf,
    }


def search_frequency(query, index, preprocess_func=None, top_k=10):
    if preprocess_func:
        query = preprocess_func(query)

    scores = defaultdict(int)

    for term in query.split():
        if term not in index:
            continue
        for doc_id, tf in index[term].items():
            scores[doc_id] += tf

    if not scores:
        return pd.Series(dtype=float)

    return pd.Series(scores).sort_values(ascending=False).head(top_k)


def search_bm25(query, index, bm25_stats, preprocess_func=None, top_k=10, k1=1.5, b=0.75):
    if preprocess_func:
        query = preprocess_func(query)

    doc_lengths = bm25_stats["doc_lengths"]
    avgdl = bm25_stats["avgdl"]
    idf = bm25_stats["idf"]

    scores = defaultdict(float)

    for term in query.split():
        if term not in index:
            continue

        for doc_id, tf in index[term].items():
            dl = doc_lengths[doc_id]

            score = idf[term] * (
                tf * (k1 + 1) /
                (tf + k1 * (1 - b + b * dl / avgdl))
            )

            scores[doc_id] += score

    if not scores:
        return pd.Series(dtype=float)

    return pd.Series(scores).sort_values(ascending=False).head(top_k)


def search_dict(query, index_type, freq_index, bm25_stats, preprocess_func=None, top_k=10):
    if index_type == "freq":
        return search_frequency(query, freq_index, preprocess_func=preprocess_func, top_k=top_k)
    elif index_type == "bm25":
        return search_bm25(query, freq_index, bm25_stats, preprocess_func=preprocess_func, top_k=top_k)
    else:
        raise ValueError("index_type must be 'freq' or 'bm25'")
