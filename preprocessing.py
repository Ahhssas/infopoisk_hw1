
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from razdel import tokenize
from pymorphy3 import MorphAnalyzer

nltk.download("stopwords", quiet=True)

russian_stopwords = set(stopwords.words("russian"))
extra_stopwords = {
    "это", "всё", "все", "очень", "который", "которая", "которые",
    "просто", "типа", "ну", "вот", "ага"
}
all_stopwords = russian_stopwords | extra_stopwords

morph = MorphAnalyzer()


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = text.replace("\\r", " ").replace("\\n", " ")
    text = text.replace("ё", "е")
    text = re.sub(r"[^а-я\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()

    tokens = [token.text for token in tokenize(text)]
    tokens = [
        token for token in tokens
        if len(token) > 2 and token not in all_stopwords
    ]

    lemmas = [morph.parse(token)[0].normal_form for token in tokens]
    lemmas = [
        lemma for lemma in lemmas
        if len(lemma) > 2 and lemma not in all_stopwords
    ]

    return " ".join(lemmas)


def preprocess_corpus(
    df,
    text_col="text",
    clean_col="text_clean",
    drop_duplicates=True,
    min_clean_words=3
):
    corpus = df.copy()

    corpus[clean_col] = corpus[text_col].fillna("").astype(str).apply(preprocess_text)
    corpus = corpus[corpus[clean_col].str.strip().astype(bool)].copy()

    corpus["clean_word_count"] = corpus[clean_col].str.split().str.len()
    corpus = corpus[corpus["clean_word_count"] >= min_clean_words].copy()

    if drop_duplicates:
        corpus = corpus.drop_duplicates(subset=[clean_col]).copy()

    corpus = corpus.reset_index(drop=True)
    return corpus
