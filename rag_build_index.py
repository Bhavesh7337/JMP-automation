import os
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import argparse

def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts)

def build_index(pdf_path: str, persist_dir: str = "app/rag_index"):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return

    os.makedirs(persist_dir, exist_ok=True)
    print(f"Reading PDF: {pdf_path}")
    raw = pdf_to_text(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = splitter.split_text(raw)
    print(f"Chunks: {len(chunks)}")

 
    api_key = "use your own api key :)"
  
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory=persist_dir)
    vectordb.persist()
    print(f" Index built at: {os.path.abspath(persist_dir)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a RAG index from a PDF file.")
    parser.add_argument("pdf_path", help="The path to the PDF file.")
    args = parser.parse_args()

    build_index(args.pdf_path)
