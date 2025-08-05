import streamlit as st
import json
import os
from datetime import datetime
from rag_pipeline import answerQuery, retrieveDocs, llm_model
from vector_database import uploadPDF, createChunks, loadPDF, getEmbeddingModel, createVectorStore
import tempfile

st.set_page_config(
    page_title="AI Lawyer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom title styling */
    .title-container {
        text-align: center;
        padding: 1rem 0 2rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 400;
    }
    
    /* Chat message styling */
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    
    .user-message {
        background-color: #3b82f6;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        margin-left: 2rem;
        text-align: right;
    }
    
    .ai-message {
        background-color: white;
        color: #374151;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        margin-right: 2rem;
        border: 1px solid #e5e7eb;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
        transition: background-color 0.2s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Input styling */
    .stTextArea > div > div > textarea {
        border-radius: 0.5rem;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 1px #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {
            'total_queries': 0,
            'documents_uploaded': 0,
            'first_visit': datetime.now().isoformat()
        }
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False


def save_chat_history():
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/chat_history.json', 'w') as f:
            json.dump(st.session_state.chat_history, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat history: {e}")


def load_chat_history():
    try:
        if os.path.exists('data/chat_history.json'):
            with open('data/chat_history.json', 'r') as f:
                st.session_state.chat_history = json.load(f)
    except Exception as e:
        st.error(f"Error loading chat history: {e}")


def save_user_data():
    try:
        os.makedirs('data', exist_ok=True)
        with open('data/user_data.json', 'w') as f:
            json.dump(st.session_state.user_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving user data: {e}")


def load_user_data():
    try:
        if os.path.exists('data/user_data.json'):
            with open('data/user_data.json', 'r') as f:
                st.session_state.user_data = json.load(f)
    except Exception as e:
        st.error(f"Error loading user data: {e}")

def process_uploaded_files(uploaded_files):
    try:
        with st.spinner("Processing uploaded documents..."):
            all_chunks = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                

                documents = loadPDF(tmp_file_path)
                chunks = createChunks(documents)
                all_chunks.extend(chunks)
                

                os.unlink(tmp_file_path)
            

            if all_chunks:
                embedding_model = getEmbeddingModel("nomic-embed-text")
                vector_store = createVectorStore(all_chunks, embedding_model, "vectorstore/temp_db_faiss")
                st.success(f"Successfully processed {len(uploaded_files)} documents with {len(all_chunks)} chunks!")
                st.session_state.documents_processed = True
                st.session_state.user_data['documents_uploaded'] += len(uploaded_files)
                save_user_data()
                return True
    except Exception as e:
        st.error(f"Error processing documents: {e}")
        return False
    
    return False


def display_chat_history():
    if st.session_state.chat_history:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in st.session_state.chat_history:
            st.markdown(f'''
                <div class="user-message">
                    <strong>You:</strong> {chat['query']}
                    <br><small>{chat['timestamp']}</small>
                </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
                <div class="ai-message">
                    <strong>⚖️ AI Lawyer:</strong> {chat['response']}
                </div>
            ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No chat history yet. Start by asking a question!")


def main():

    init_session_state()
    load_chat_history()
    load_user_data()
    
    # Header
    st.markdown('''
        <div class="title-container">
            <div class="title-text">⚖️ AI Lawyer</div>
            <div class="subtitle-text">Your Intelligent Legal Assistant for Human Rights Law</div>
        </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.subheader("📊 Your Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", st.session_state.user_data['total_queries'])
        with col2:
            st.metric("Documents Uploaded", st.session_state.user_data['documents_uploaded'])
        
        st.divider()
        
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload legal documents to enhance the AI's knowledge base"
        )
        
        if uploaded_files and st.button("Process Documents", type="primary"):
            process_uploaded_files(uploaded_files)
        
        st.divider()
        
        st.subheader("🗂️ Actions")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            save_chat_history()
            st.rerun()
        
        if st.button("Download Chat History"):
            if st.session_state.chat_history:
                chat_json = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=chat_json,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with AI Lawyer")
        
        # Display chat history
        display_chat_history()
        
        # Query input
        user_query = st.text_area(
            "Ask your legal question:",
            height=120,
            placeholder="e.g., 'What are the fundamental human rights according to Article 18?'",
            help="Ask questions about human rights law, legal procedures, or constitutional matters."
        )
        
        # Submit button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            ask_question = st.button("Ask AI Lawyer", type="primary")
    
    with col2:
        st.subheader("📋 Quick Help")
        
        with st.expander("🔍 Sample Questions"):
            st.markdown("""
            - What rights are guaranteed under Article 19?
            - If a government forbids peaceful assembly, which articles are violated?
            - What does the UDHR say about freedom of expression?
            - Which articles protect against discrimination?
            - What are the limitations on human rights?
            """)
        
        with st.expander("💡 Tips"):
            st.markdown("""
            - Be specific in your questions
            - Reference article numbers when possible
            - Ask about violations and remedies
            - Inquire about legal procedures
            - Upload relevant documents for better context
            """)
        
        with st.expander("⚠️ Disclaimer"):
            st.markdown("""
            This AI assistant provides information for educational purposes only. 
            For actual legal advice, please consult qualified legal professionals.
            """)
    
    # Process query
    if ask_question:
        if uploaded_files and not st.session_state.documents_processed:
            processed = process_uploaded_files(uploaded_files)
            if not processed:
                st.error("Failed to process uploaded documents. Please try again.")
                return
        if user_query.strip():
            try:
                with st.spinner("🔍 Analyzing your question..."):
                    retrieved_docs = retrieveDocs(user_query)
                    answer = answerQuery(documents=retrieved_docs, model=llm_model, query=user_query)
                    chat_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': user_query,
                        'response': answer,
                        'documents_count': len(retrieved_docs)
                    }
                    st.session_state.chat_history.append(chat_entry)
                    st.session_state.user_data['total_queries'] += 1
                    save_chat_history()
                    save_user_data()
                    st.session_state.user_query = ""
                    st.rerun()
            except Exception as e:
                st.error(f"Error processing your question: {e}")
                st.info("Please make sure the vector database is properly set up and Ollama is running.")
        else:
            st.warning("Please enter a question before asking the AI Lawyer.")
            st.session_state.user_query = ""
    
    elif ask_question and not user_query.strip():
            st.warning("Please enter a question before asking the AI Lawyer.")
            st.session_state.user_query = ""
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6b7280; font-size: 0.9rem;'>"
        "For educational & research purposes only"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()