"""Text processing utilities for the RAG application."""

import re
from typing import List

import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")


def clean_text(text: str) -> str:
    """Clean and preprocess text.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep punctuation
    text = re.sub(r"[^\w\s.,!?;:]", "", text)

    # Convert to lowercase
    text = text.lower()

    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    return sent_tokenize(text)


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text.

    Args:
        text: Input text

    Returns:
        Text with stopwords removed
    """
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


def preprocess_text(text: str) -> str:
    """Apply all preprocessing steps to text.

    Args:
        text: Input text

    Returns:
        Preprocessed text
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    return text
