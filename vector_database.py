import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Configuration
PDFS_DIRECTORY = 'pdfs/'
FAISS_DB_PATH = "vectorstore/db_faiss"
OLLAMA_MODEL_NAME = "nomic-embed-text"  # Better for embeddings

def uploadPDF(file):
    """Upload PDF file to the pdfs directory"""
    os.makedirs(PDFS_DIRECTORY, exist_ok=True)
    file_path = os.path.join(PDFS_DIRECTORY, file.name)
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def loadPDF(filePath):
    """Load PDF document using PDFPlumberLoader"""
    if not os.path.exists(filePath):
        raise FileNotFoundError(f"PDF file not found: {filePath}")
    
    loader = PDFPlumberLoader(filePath)
    documents = loader.load()
    return documents

def createChunks(documents, chunk_size=1000, chunk_overlap=200):
    """Create text chunks from documents"""
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    textChunks = textSplitter.split_documents(documents)
    return textChunks

def getEmbeddingModel(model_name):
    """Initialize Ollama embedding model"""
    try:
        print(f"Initializing embedding model: {model_name}")
        embeddings = OllamaEmbeddings(model=model_name)
        
        # Test the model with a simple embedding
        test_embedding = embeddings.embed_query("test")
        print(f"Model initialized successfully. Embedding dimension: {len(test_embedding)}")
        
        return embeddings
    except Exception as e:
        print(f"Error initializing embedding model '{model_name}': {e}")
        print("Available solutions:")
        print(f"1. Pull the model: ollama pull {model_name}")
        print("2. Use a different model like 'nomic-embed-text' or 'all-minilm'")
        print("3. Check available models: ollama list")
        raise

def createVectorStore(text_chunks, embedding_model, db_path):
    """Create and save FAISS vector store"""
    try:
        # Create FAISS vector store
        faiss_db = FAISS.from_documents(text_chunks, embedding_model)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Save the vector store
        faiss_db.save_local(db_path)
        print(f"Vector store saved successfully at: {db_path}")
        return faiss_db
    except Exception as e:
        print(f"Error creating vector store: {e}")
        raise

def loadVectorStore(db_path=None, model_name=None):
    """Load existing FAISS vector store from disk"""
    if db_path is None:
        db_path = FAISS_DB_PATH
    if model_name is None:
        model_name = OLLAMA_MODEL_NAME
    
    try:
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Vector store not found at: {db_path}")
        
        embedding_model = getEmbeddingModel(model_name)
        faiss_db = FAISS.load_local(
            db_path, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        print(f"Vector store loaded successfully from: {db_path}")
        return faiss_db
    except Exception as e:
        print(f"Error loading vector store: {e}")
        raise

def main():
    """Main function to process PDF and create vector store"""
    try:
        # Load PDF
        file_path = 'universal_declaration_of_human_rights.pdf'
        print(f"Loading PDF: {file_path}")
        documents = loadPDF(file_path)
        print(f"PDF Pages: {len(documents)}")
        
        # Create chunks
        print("Creating text chunks...")
        text_chunks = createChunks(documents)
        print(f"Chunks Count: {len(text_chunks)}")
        
        # Initialize embedding model
        print(f"Initializing embedding model: {OLLAMA_MODEL_NAME}")
        embedding_model = getEmbeddingModel(OLLAMA_MODEL_NAME)
        
        # Create and save vector store
        print("Creating vector store...")
        vector_store = createVectorStore(text_chunks, embedding_model, FAISS_DB_PATH)
        
        print("Process completed successfully!")
        return vector_store
        
    except Exception as e:
        print(f"Error in main process: {e}")
        raise

# Create a global variable that can be imported
faiss_db = None

def get_faiss_db():
    """Get or load the FAISS database"""
    global faiss_db
    if faiss_db is None:
        faiss_db = loadVectorStore()
    return faiss_db

if __name__ == "__main__":
    main()