
import argparse
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

    show_cols = [col for col in ["theme", text_col, "rating", "score"] if col in result.columns]
    print(result[show_cols].head(top_k).to_string(index=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--backend",
        default="dict_bm25",
        choices=["dict_freq", "dict_bm25", "matrix_freq", "matrix_bm25"]
    )
    parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()

    df = pd.read_csv(args.input)
    clean_df = preprocess_corpus(df, text_col=args.text_col)

    docs, tokenized_docs = prepare_tokenized_docs(clean_df, clean_col="text_clean")

    freq_index = build_freq_inverted_index(tokenized_docs)
    bm25_stats = build_bm25_stats(freq_index, tokenized_docs)

    tf_matrix, feature_names, vocabulary = build_tf_matrix(tokenized_docs)
    bm25_matrix, bm25_matrix_stats = build_bm25_matrix(tf_matrix)

    if args.backend == "dict_freq":
        scores = search_frequency(args.query, freq_index, preprocess_func=preprocess_text, top_k=args.top_k)
    elif args.backend == "dict_bm25":
        scores = search_bm25(args.query, freq_index, bm25_stats, preprocess_func=preprocess_text, top_k=args.top_k)
    elif args.backend == "matrix_freq":
        scores = search_freq_matrix(args.query, tf_matrix, vocabulary, preprocess_func=preprocess_text, top_k=args.top_k)
    elif args.backend == "matrix_bm25":
        scores = search_bm25_matrix(args.query, bm25_matrix, vocabulary, preprocess_func=preprocess_text, top_k=args.top_k)
    else:
        raise ValueError("Unknown backend")

    print_results(clean_df, scores, top_k=args.top_k, text_col=args.text_col)


if __name__ == "__main__":
    main()
