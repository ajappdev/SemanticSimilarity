# APP IMPORTATIONS
import streamlit as st
import docx
import PyPDF2
import tempfile
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

def read_docx(file):
    """Reads text content from a DOCX file."""
    doc = docx.Document(file)
    text = [paragraph.text for paragraph in doc.paragraphs]
    return '\n'.join(text)

def read_pdf(file):
    """Reads text content from a PDF file."""
    temp_pdf = tempfile.NamedTemporaryFile(delete=False)
    temp_pdf.write(file.read())
    temp_pdf_path = temp_pdf.name
    temp_pdf.close()

    with open(temp_pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfFileReader(f)
        text = [pdf_reader.getPage(i).extractText() for i in range(pdf_reader.numPages)]

    return '\n'.join(text)

def get_sentence_embeddings(sentences, model_name):
    """Generates sentence embeddings using the model 
    distilbert-base-nli-stsb-mean-tokens from Hugging Face"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences)
    return embeddings

def calculate_cosine_semantic_similarity(embeddings_1, embeddings_2):
    """
    This function calculates the cosine semantic similarity between two
    vector embedings
    """
    # Normalize the embeddings to unit length (L2 normalization)
    embeddings_1_normalized = embeddings_1 / np.linalg.norm(embeddings_1, axis=1)[:, np.newaxis]
    embeddings_2_normalized = embeddings_2 / np.linalg.norm(embeddings_2, axis=1)[:, np.newaxis]

    # Calculate the cosine similarity
    similarity_score = 1 - cosine(embeddings_1_normalized.mean(axis=0), embeddings_2_normalized.mean(axis=0))

    return similarity_score

def main():
    """
    This is the main streamlit app
    """
    ##################################################
    # Sidebar (Title, explanation of semantic similarity, upload files)
    ##################################################
    st.title("SemanticApp")
    st.text("""This App computes the semantic similarity between two texts""")
    st.text("""Developed by Ajbar Alae - February 2024""")
    st.sidebar.title("About Semantic Similarity")
    st.sidebar.write("""Semantic similarity between two texts means how much they talk about the same things.
    It's not just about having the same words but understanding the meaning behind them. 
    Imagine comparing two movie reviews - If they both talk about the same plot, characters, and emotions, 
    they have high semantic similarity. Techniques like word and sentence embeddings help computers 
    understand these meanings and figure out how similar or different texts are beyond just the words they use.""")
    
    st.sidebar.title("Let's try this!")
    
    # Upload widgets for two files
    uploaded_file_1 = st.sidebar.file_uploader("Upload your first text document (PDF or DOC format)", type=["pdf", "docx"])
    uploaded_file_2 = st.sidebar.file_uploader("Upload your second text document (PDF or DOC format)", type=["pdf", "docx"])
    
    ##################################################
    # Right Part of the screen (Semantic Similarity Results)
    ##################################################

    if st.sidebar.button("Compute Similarity"):
        if uploaded_file_1 and uploaded_file_2:
            try:
                if uploaded_file_1.type == "application/pdf":
                    content_1 = read_pdf(uploaded_file_1)
                elif uploaded_file_1.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    content_1 = read_docx(uploaded_file_1)

                if uploaded_file_2.type == "application/pdf":
                    content_2 = read_pdf(uploaded_file_2)
                elif uploaded_file_2.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    content_2 = read_docx(uploaded_file_2)

                # Check if file 1 and file 2 are not identic
                if content_1 == content_2:
                    st.warning("The two provided texts are identical. Please reupload different files and try again.")
                else:
                    st.subheader("Document 1: " + uploaded_file_1.name)
                    st.text_area("Content", content_1, height=200)

                    st.subheader("Document 2: " + uploaded_file_2.name)
                    st.text_area("Content", content_2, height=200)

                    # Upon clicking on the button, start the process and show a spinner for the user
                    with st.spinner("Calculating the semantic similarity of the two texts..."):
                        
                        sentences_1 = content_1.split('\n')
                        sentences_2 = content_2.split('\n')

                        # Convert the content of the two texts to vector embeddings
                        model_name = 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'
                        embeddings_1 = get_sentence_embeddings(sentences_1, model_name)
                        embeddings_2 = get_sentence_embeddings(sentences_2, model_name)

                        similarity_score = calculate_cosine_semantic_similarity(
                            embeddings_1,
                            embeddings_2
                        )

                    # Determine the color based on similarity score
                    color = "green" if similarity_score > 0.7 else ("black" if similarity_score > 0.4 else "red")

                    st.success("Success! Processing is complete.")

                    # Display the similarity score with appropriate color
                    st.markdown(f"<p style='font-size:36px;color:{color};font-weight:bold;'>Similarity Score: {similarity_score:.2%}</p>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}. Please make sure the uploaded files are valid and different.")

if __name__ == "__main__":
    main()
