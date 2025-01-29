import streamlit as st
import PyPDF2

st.set_page_config(page_title="Deep dive AI")

st.title("Welcome to Deep Dive AI 🚀")
st.subheader("In-depth insights from AI Research papers")

def main():
    #st.title("PDF Uploader")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Read the PDF file
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        num_pages = len(pdf_reader.pages)
        st.write(f"Number of pages: {num_pages}")

        # Display the content of the first page
        page = pdf_reader.pages[0]
        text = page.extract_text()
        st.write(text)

if __name__ == "__main__":
    main()