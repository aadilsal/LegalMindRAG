import streamlit as st
from rag_pipeline import answerQuery, retrieveDocs,llm_model

uploaded_file=st.file_uploader("Upload PDF",type="pdf",accept_multiple_files=True)

user_query=st.text_area("Enter your prompt:",height=150,placeholder="Ask me Anything!")
ask_question=st.button("Ask AI Lawyer")

if ask_question:
    
    if uploaded_file:
        st.chat_message("user").write(user_query)
        #RAG PIPLE
        # Retrieve relevant documents
        retrieved_docs = retrieveDocs(user_query)
        print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Generate answer
        print("\nGenerating answer...")
        answer = answerQuery(documents=retrieved_docs, model=llm_model, query=user_query)
        
        print("\n" + "="*50)
        print("AI LAWYER RESPONSE:")
        print("="*50)
        print(answer)
        
        # fixed_resp="HARD CODED RESPONSE"
        st.chat_message("AI Lawyer").write(answer)
    else:
        st.error("Upload a valid PDF!")