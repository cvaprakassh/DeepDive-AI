import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
import nltk
from nltk.tokenize import sent_tokenize


# Ensure nltk punkt resource is downloaded
nltk.download('punkt')

st.set_page_config(page_title="Deep dive AI")

st.title("Welcome to Deep Dive AI 🚀")
st.subheader("In-depth insights from AI Research papers")

st.write("Deep Dive AI is a tool that helps you to extract information from research papers using AI. Upload a PDF file and ask a question to get started!")

def load_huggingface_model(prompt):
    # Load the model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad_token as eos_token to prevent padding issues
    tokenizer.pad_token = tokenizer.eos_token  

    # Tokenize the input text
    inputs = tokenizer(prompt, return_tensors='pt', max_length=1024, truncation=True, padding=True)
    
    # Create a text-generation pipeline
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    # Generate text
    generated = generator(prompt, max_new_tokens=150, num_return_sequences=1)
    
    # Extract the generated text correctly (from the 'generated_text' key in the output dictionary)
    generated_text = generated[0]['generated_text']
    
    return generated_text




def extract_text_from_pdf(uploaded_file):
    
   # Read the PDF file
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    num_pages = len(pdf_reader.pages)
    st.write(f"Number of pages: {num_pages}")
    text=""

    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def preprocess_text(text):
    # Split the text into paragraphs based on double newlines (i.e., paragraphs are separated by two line breaks)
    #paragraphs = text.split('\n\n')
    
    # Clean up each paragraph by stripping extra spaces or other preprocessing tasks
    #paragraphs = [para.strip() for para in paragraphs if para.strip() != ""]
    sentences = sent_tokenize(text)

    
    return sentences


#generate embeddings for each paragraph
def generate_embeddings(paragraphs,batch_size=32):
    model= SentenceTransformer('all-MiniLM-L6-v2') #load the model
    embeddings = [] #initialize an empty list to store the embeddings
    # Generate embeddings in batches
    for i in range(0, len(paragraphs), batch_size):
        batch = paragraphs[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True) #generate embeddings
        embeddings.append(batch_embeddings.cpu().numpy()) #append the embeddings to the list
    # Ensure embeddings are on CPU and convert to numpy
    #if embeddings.device.type == 'mps':  # If the tensor is on MPS (Apple Silicon GPU)
    #   embeddings = embeddings.cpu()  # Move it to CPU
    #embeddings_numpy=embeddings.detach().numpy() #convert embeddings to numpy array


    return np.vstack(embeddings) #return the embeddings as a numpy array


#indexing with FAISS
def index_faiss(embeddings):

    #embeddings=np.array(embeddings).astype('float32') #convert embeddings to numpy array
    index=faiss.IndexFlatL2(embeddings.shape[1]) #create a Faiss index for L2 distance
    index.add(embeddings) #add the embeddings to the index
    return index

#searching with FAISS
def search_faiss(query, index, paragraphs,model, k=5):
    query_embedding = model.encode([query]).astype('float32')     #generate the query embedding
    D, I = index.search(query_embedding, k) #search the index for the nearest neighbors
    results = [(paragraphs[i], D[0][j]) for j, i in enumerate(I[0])] #get the paragraphs and distances of the nearest neighbors
    return results


uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
text=''
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    #st.write(text)
    paragraphs = preprocess_text(text)
    embeddings = generate_embeddings(paragraphs)
    index = index_faiss(embeddings)
    model= SentenceTransformer('all-MiniLM-L6-v2')
    query = st.text_input("Enter your question here:")
    if query == "":
        st.write("Please enter a question to get started")
    else:
        results = search_faiss(query, index, paragraphs, model,k=5)
        joined_results = ''.join([text for text, _ in results])  # This works because you are extracting the string part

        #using Hugging Face model to generate answers
        prompt = f"Answer the following question using the provided information:\n\nQuestion: {query}\n\nContext:\n" + "\n".join(joined_results) + "\n\nAnswer:"
        #st.write("Prompt:", prompt)
        #if st.button("Generate Answer"):
        generated_answer = load_huggingface_model(prompt)
        st.write("Generated Answer:")
        st.write(generated_answer)
else:
    st.write("Please upload a PDF file to get started")
