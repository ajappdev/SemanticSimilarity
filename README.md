# SemanticApp

SemanticApp is a web application that computes the semantic similarity between two text documents. Semantic similarity measures how much two texts discuss the same topics, considering the meaning behind the words rather than just their presence. The application utilizes state-of-the-art natural language processing techniques and models to provide accurate results.

## Table of Contents

- [Semantic Similarity](#semantic-similarity)
- [Purpose](#purpose)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [Author](#author)

## Semantic Similarity

Semantic similarity refers to the degree of likeness between two pieces of text in terms of their meaning. It goes beyond simple word matching and considers the context and understanding of the content. In the context of SemanticApp, the application calculates the semantic similarity between two uploaded text documents.

## Purpose

The purpose of SemanticApp is to provide users with a tool for assessing how closely related two pieces of text are in terms of content and meaning. This can be useful in various applications such as document comparison, plagiarism detection, and content recommendation.

## Technologies Used

SemanticApp utilizes the following technologies:

- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity): A measure of similarity between two non-zero vectors.
- [DistilBERT](https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens): A pretrained transformer model for natural language understanding.
- [Streamlit](https://streamlit.io/): A Python library for creating interactive web applications.
- [Python](https://www.python.org/): The programming language used for building the application.
- [NumPy](https://numpy.org/): A library for numerical operations in Python.
- Vector Embeddings: Representations of text in a high-dimensional space used for semantic analysis.

## How It Works

1. **Upload Documents**: Users upload two text documents (in PDF or DOCX format) through the web interface.

2. **Document Processing**: The application reads the content of the documents using specialized functions for DOCX and PDF formats.

3. **Semantic Embeddings**: The text content is converted into vector embeddings using the DistilBERT model. These embeddings capture the semantic meaning of the text.

4. **Cosine Similarity**: Cosine similarity is calculated between the normalized embeddings of the two documents. This yields a semantic similarity score.

5. **User Feedback**: The application displays the content of the documents and the computed similarity score. The score is color-coded based on its magnitude, providing an intuitive understanding of the relationship between the texts.

## Author

SemanticApp is developed and maintained by Ajbar Alae in February 2024. Feel free to reach out for questions, feedback, or contributions at alae1ajbar@gmail.com

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
