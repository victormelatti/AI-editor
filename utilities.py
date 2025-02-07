from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def extract_text_from_pdfs(pdf_paths, faiss_index_path):
    """Loads, extracts text from multiple PDFs, and saves embeddings if not already stored."""
    # Check if FAISS index exists
    if os.path.exists(faiss_index_path):
        print("✅ FAISS index found. Loading existing embeddings...")
        return FAISS.load_local(faiss_index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    print("🛠️ FAISS index not found. Creating embeddings...")

    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)  # Collect all documents

    # Chunk text into smaller pieces for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(all_docs)

    # Convert to FAISS vector database
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    # Save FAISS index
    vectorstore.save_local(faiss_index_path)
    print(f"✅ FAISS index saved at: {faiss_index_path}")

    return vectorstore

# ✅ Multi-PDF Retrieval Tool
def retrieve_pdf_info(query, pdfs_paths, faiss_index_path):
    """Retrieves relevant information from multiple PDFs using saved FAISS embeddings."""
    vectorstore = extract_text_from_pdfs(pdfs_paths, faiss_index_path)
    docs = vectorstore.similarity_search(query, k=5)  # Retrieve top N results

    return "\n\n".join([doc.page_content for doc in docs])

def format_text(text, words_per_line=30):
    """Breaks text into lines of a specified word limit."""
    words = text.split()
    formatted_text = "\n".join(
        [" ".join(words[i:i+words_per_line]) for i in range(0, len(words), words_per_line)]
    )
    return formatted_text