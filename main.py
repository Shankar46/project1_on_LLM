import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Suppress insecure request warnings
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

def scrape_html(url):
    scraped_chunks = []
    try:
        with requests.get(url, verify=False) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(strip=True)
            cleaned_text = clean_text(text_content) 
            st.write("\nMain Page Text content:")
            st.write(cleaned_text)  # Print main page content for verification
            scraped_chunks.append(cleaned_text)
            # Find and scrape links within the main page
            links = soup.find_all('a', href=True)
            if links:
                for link in links:
                    link_url = link.get('href')
                    if is_valid_link(link_url):
                        st.write(f"Scraping linked page: {link_url}")
                        scrape_linked_page(link_url, scraped_chunks)
            process_chunks(scraped_chunks)
    except requests.RequestException as e:
        st.error(f"Error: Could not retrieve HTML content from {url}. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while scraping {url}. Error: {e}")

def scrape_linked_page(link_url, scraped_chunks):
    try:
        with requests.get(link_url, verify=False) as response:
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text(strip=True)
            cleaned_text = clean_text(text_content)
            st.write("\nLinked Page Text content:")
            st.write(cleaned_text)  # Print linked page content for verification
            scraped_chunks.append(cleaned_text)
            # Find and scrape links within the linked page
            links = soup.find_all('a', href=True)
            if links:
                for link in links:
                    inner_link_url = link.get('href')
                    if is_valid_link(inner_link_url):
                        st.write(f"Scraping linked page: {inner_link_url}")
                        scrape_linked_page(inner_link_url, scraped_chunks)
    except requests.RequestException as e:
        st.error(f"Error: Could not retrieve HTML content from {link_url}. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while scraping {link_url}. Error: {e}")

def clean_text(text):
    # Keep only English language characters, numbers, and common punctuation marks
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    cleaned_text = re.sub(r'[^\w\s.,]', '', cleaned_text)
    return cleaned_text

def process_chunks(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_db = FAISS.from_texts(chunks, embeddings)
    faiss_db.save_local("faiss_index")
    st.write("\nProcessed and saved chunks.")

def get_conversation_chain():
    prompt_template = """ 
    Answer the question as detailed as possible based on the text content scraped from the web.\n
    If the answer is not available in the scraped content, please indicate so.\n\n
    Context:\n
    {context}\n\n
    Question:\n
    {question}\n\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    print("Documents returned by similarity search:", docs)  # Print returned documents
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print("Conversation Chain Output:", response)  # Print conversation chain output
    st.write("Reply: ", response["output_text"])

def is_valid_link(link_url):
    # Check if the link URL is not a document, image, Gmail link, or contact number
    if (not re.match(r'.+\.(pdf|docx?|xlsx?|pptx?|jpg|jpeg|png|gif)', link_url, re.IGNORECASE) and 
        not 'mail.google.com' in link_url and 
        not re.match(r'tel:\d+', link_url)):
        return True
    else:
        return False

def main():
    st.set_page_config("Scrapped Data")
    st.header("Chat with Scrapped Data using Gemini")
    user_question = st.text_input("Ask a Question:")
    if st.button("Enter") and user_question:
        user_input(user_question)
    target_url = st.text_input("Enter the website link")   
    if st.button("Scrape HTML") and target_url:
        scrape_html(target_url)

if __name__ == "__main__":
    main()
