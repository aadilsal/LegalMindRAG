from langchain_groq import ChatGroq
from vector_database import get_faiss_db  # Import the function instead
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the language model
llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

def retrieveDocs(query, k=4):
    """Retrieve relevant documents from the vector database"""
    try:
        faiss_db = get_faiss_db()  # Get the FAISS database
        return faiss_db.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        raise

def getContext(documents):
    """Extract context from retrieved documents"""
    context = "\n\n".join([doc.page_content for doc in documents])
    return context

# Custom prompt template for the RAG system
customPromptTemplate = """
Use the pieces of information provided in the context to answer the user's question about human rights law. 
If you don't know the answer based on the provided context, just say that you don't know - don't try to make up an answer. 
Only provide information that can be found in the given context.

Question: {question}
Context: {context}

Answer:
"""

def answerQuery(documents, model, query):
    """Generate answer using retrieved documents and language model"""
    try:
        context = getContext(documents)
        prompt = ChatPromptTemplate.from_template(customPromptTemplate)
        chain = prompt | model
        
        response = chain.invoke({"question": query, "context": context})
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"Error generating answer: {e}")
        raise

def main():
    """Main function to run the RAG pipeline"""
    try:
        # Test question
        question = "If a government forbids the right to assemble peacefully, which articles are violated and why?"
        
        print(f"Question: {question}")
        print("\nRetrieving relevant documents...")
        
        # Retrieve relevant documents
        retrieved_docs = retrieveDocs(question)
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Generate answer
        print("\nGenerating answer...")
        answer = answerQuery(documents=retrieved_docs, model=llm_model, query=question)
        
        print("\n" + "="*50)
        print("AI LAWYER RESPONSE:")
        print("="*50)
        print(answer)
        
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        raise

if __name__ == "__main__":
    main()