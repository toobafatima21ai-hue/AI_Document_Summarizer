from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

from file_handler import load_txt, load_pdf
from preprocess import preprocess_text
from summarizer import frequency_summary, tfidf_summary
from analytics import word_frequency, top_keywords
from pdf_export import create_pdf

from langdetect import detect

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Summarizer",
    layout="wide"
)

st.title("AI-Powered Document Summarization System")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload One or More Documents",
    type=["txt", "pdf"],
    accept_multiple_files=True
)

text_input = st.text_area(
    "Or Paste Text Here",
    height=250
)

# --------------------------------------------------
# SUMMARIZATION METHOD
# --------------------------------------------------

method = st.selectbox(
    "Summarization Method",
    [
        "Frequency Based",
        "TF-IDF Based"
    ]
)

summary_length = st.slider(
    "Summary Length (%)",
    min_value=10,
    max_value=90,
    value=30
)

# --------------------------------------------------
# READ DOCUMENTS
# --------------------------------------------------

text = ""

if uploaded_files:

    for file in uploaded_files:

        try:

            if file.name.lower().endswith(".txt"):

                text += load_txt(file)

            elif file.name.lower().endswith(".pdf"):

                text += load_pdf(file)

            text += "\n\n"

        except Exception as e:

            st.error(
                f"Error reading {file.name}: {e}"
            )

elif text_input:

    text = text_input

# --------------------------------------------------
# GENERATE SUMMARY
# --------------------------------------------------

if st.button("Generate Summary"):

    if not text.strip():

        st.error(
            "Please upload document(s) or enter text."
        )

    else:

        try:

            # ------------------------------------------
            # PREPROCESSING
            # ------------------------------------------

            words, sentences = preprocess_text(text)

            # ------------------------------------------
            # LANGUAGE DETECTION
            # ------------------------------------------

            language = detect(text)

            # ------------------------------------------
            # SUMMARIZATION
            # ------------------------------------------

            if method == "Frequency Based":

                summary, scores = frequency_summary(
                    text,
                    summary_length
                )

            else:

                summary, scores = tfidf_summary(
                    text,
                    summary_length
                )

            # ------------------------------------------
            # ORIGINAL VS SUMMARY
            # ------------------------------------------

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Original Text")

                st.write(text)

            with col2:

                st.subheader("Summary")

                st.write(summary)

            # ------------------------------------------
            # DOCUMENT STATISTICS
            # ------------------------------------------

            st.subheader("Document Statistics")

            st.success(
                f"Detected Language: {language}"
            )

            original_words = len(text.split())

            summary_words = len(summary.split())

            compression = round(
                (
                    1 -
                    summary_words /
                    original_words
                ) * 100,
                2
            )

            stat1, stat2, stat3 = st.columns(3)

            stat1.metric(
                "Original Words",
                original_words
            )

            stat2.metric(
                "Summary Words",
                summary_words
            )

            stat3.metric(
                "Compression %",
                compression
            )

            # ------------------------------------------
            # WORD FREQUENCY ANALYSIS
            # ------------------------------------------

            st.subheader(
                "Word Frequency Analysis"
            )

            freq_df = pd.DataFrame(
                word_frequency(words),
                columns=[
                    "Word",
                    "Frequency"
                ]
            )

            st.dataframe(freq_df)

            st.bar_chart(
                freq_df.set_index("Word")
            )

            # ------------------------------------------
            # WORD CLOUD
            # ------------------------------------------

            st.subheader("Word Cloud")

            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white"
            ).generate(
                " ".join(words)
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.imshow(wordcloud)

            ax.axis("off")

            st.pyplot(fig)

            # ------------------------------------------
            # IMPORTANT KEYWORDS
            # ------------------------------------------

            st.subheader(
                "Important Keywords"
            )

            keyword_df = pd.DataFrame(
                top_keywords(words),
                columns=[
                    "Keyword",
                    "Score"
                ]
            )

            st.dataframe(keyword_df)

            # ------------------------------------------
            # SENTENCE IMPORTANCE SCORES
            # ------------------------------------------

            st.subheader(
                "Sentence Importance Scores"
            )

            score_df = pd.DataFrame(
                list(scores.items()),
                columns=[
                    "Sentence",
                    "Score"
                ]
            )

            st.dataframe(score_df)

                         # ------------------------------------------
            # DOWNLOAD TXT
            # ------------------------------------------

            st.download_button(
                label="Download Summary (.txt)",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )

            # ------------------------------------------
            # DOWNLOAD PDF
            # ------------------------------------------

            pdf_file = create_pdf(
                summary,
                "summary.pdf"
            )

            with open(pdf_file, "rb") as file:

                st.download_button(
                    label="Download Summary (.pdf)",
                    data=file,
                    file_name="summary.pdf",
                    mime="application/pdf"
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )