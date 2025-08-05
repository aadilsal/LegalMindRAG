from langchain_groq import ChatGroq
from vector_database import get_faiss_db 
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()


llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b")

def retrieveDocs(query, k=4):
    try:
        faiss_db = get_faiss_db()  
        return faiss_db.similarity_search(query, k=k)
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        raise

def getContext(documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    return context


customPromptTemplate = """
Use the pieces of information provided in the context to answer the user's question about human rights law. 
If you don't know the answer based on the provided context, just say that you don't know - don't try to make up an answer. 
Only provide information that can be found in the given context.

Question: {question}
Context: {context}

Answer:
"""

def answerQuery(documents, model, query):
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
    try:
        question = "If a government forbids the right to assemble peacefully, which articles are violated and why?"
        
        print(f"Question: {question}")
        print("\nRetrieving relevant documents...")
        
        retrieved_docs = retrieveDocs(question)
        print(f"Retrieved {len(retrieved_docs)} documents")
        
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