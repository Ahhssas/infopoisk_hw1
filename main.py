
import argparse
from pathlib import Path

import pandas as pd

from preprocessing import preprocess_corpus, preprocess_text
from dict_indexes import (
    prepare_tokenized_docs,
    build_freq_inverted_index,
    build_bm25_stats,
    search_frequency,
    search_bm25,
)
from matrix_indexes import (
    build_tf_matrix,
    build_bm25_matrix,
    search_freq_matrix,
    search_bm25_matrix,
)


def print_results(df, scores, top_k=10, text_col="text"):
    if scores.empty:
        print("Ничего не найдено.")
        return

    result = df.iloc[scores.index].copy()
    result["score"] = scores.values

    cols = [c for c in ["theme", text_col, "rating", "score"] if c in result.columns]
    print(result[cols].head(top_k).to_string(index=True))


def run_search(
    input_path="jokes_1500.csv",
    text_col="text",
    query="муж жена",
    backend="dict_bm25",
    top_k=10,
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    df = pd.read_csv(input_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in dataset.")

    clean_df = preprocess_corpus(df, text_col=text_col)
    _, tokenized_docs = prepare_tokenized_docs(clean_df, clean_col="text_clean")

    freq_index = build_freq_inverted_index(tokenized_docs)
    bm25_stats = build_bm25_stats(freq_index, tokenized_docs)
    tf_matrix, _, vocabulary = build_tf_matrix(tokenized_docs)
    bm25_matrix, _ = build_bm25_matrix(tf_matrix)

    if backend == "dict_freq":
        scores = search_frequency(query, freq_index, preprocess_func=preprocess_text, top_k=top_k)
    elif backend == "dict_bm25":
        scores = search_bm25(query, freq_index, bm25_stats, preprocess_func=preprocess_text, top_k=top_k)
    elif backend == "matrix_freq":
        scores = search_freq_matrix(query, tf_matrix, vocabulary, preprocess_func=preprocess_text, top_k=top_k)
    elif backend == "matrix_bm25":
        scores = search_bm25_matrix(query, bm25_matrix, vocabulary, preprocess_func=preprocess_text, top_k=top_k)
    else:
        raise ValueError("Unknown backend")

    print_results(clean_df, scores, top_k=top_k, text_col=text_col)
    return scores


def main():
    parser = argparse.ArgumentParser(description="Search over a jokes corpus.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--backend",
        default="dict_bm25",
        choices=["dict_freq", "dict_bm25", "matrix_freq", "matrix_bm25"],
    )
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    run_search(
        input_path=args.input,
        text_col=args.text_col,
        query=args.query,
        backend=args.backend,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
