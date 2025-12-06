import streamlit as st
import pandas as pd
import random
import time
from typing import List, Dict, Any

# --- New Imports for Gemini API and Pinecone ---
try:
    from google import genai
    from google.genai import types
    from pinecone import Pinecone, ApiException
except ImportError:
    # Set to None if imports fail, triggering the error messages later in the script
    genai = None
    Pinecone = None
    ApiException = None


# --- Configuration and Initialization ---

# 1. Page Configuration
st.set_page_config(
    page_title="UCSC Campus Resource Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. UCSC Resource Assistant (Actual RAG Implementation)
class UCSCResourceAssistant:
    """Handles retrieval (Pinecone) and answer generation (Gemini API)."""
    
    def __init__(self, gemini_client, pinecone_index, embedding_model):
        self.vector_store_ready = True
        self.gemini_client = gemini_client
        self.pinecone_index = pinecone_index
        self.embedding_model = embedding_model
        self.model_name = "gemini-2.5-flash"

    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Performs a vector search against the Pinecone index."""
        if not self.vector_store_ready:
            # Should be handled during initialization, but kept as a safeguard
            raise ConnectionError("Vector database is unavailable.")

        try:
            # 1. Create the query embedding using Gemini
            query_embedding_response = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            query_vector = query_embedding_response['embedding']

            # 2. Search Pinecone for relevant document vectors
            search_results = self.pinecone_index.query(
                vector=query_vector,
                top_k=3, # Retrieve top 3 most relevant documents
                include_metadata=True
            )
            
            # 3. Format Pinecone results into the required snippet format
            retrieved_docs = []
            for match in search_results.matches:
                # Assume metadata stores the original text ('text') and source ('source')
                if 'text' in match.metadata and 'source' in match.metadata:
                    retrieved_docs.append({
                        "source": match.metadata['source'], 
                        "snippet": match.metadata['text']
                    })
            
            return retrieved_docs

        except ApiException as e:
            # Raise a custom error for better user feedback
            raise ConnectionError(f"Pinecone API Error during search: {e}")
        except Exception as e:
            # Catch general errors (like embedding generation failure)
            st.warning(f"Embedding or Pinecone search failed: {e}")
            return []


    def generate_answer(self, query: str, documents: List[Dict[str, Any]], style: str) -> str:
        """Generates LLM answer using the Gemini API, grounded by retrieved documents."""
        
        # 1. Format the retrieved context for the LLM
        context = "\n---\n".join([f"Source: {d['source']}\nSnippet: {d['snippet']}" for d in documents])
        
        if not context:
            return "I couldn't find any relevant campus resources in the indexed documents. Please try a different query."

        # 2. Define the system instruction (RAG prompt)
        system_instruction = (
            "You are a helpful, verified campus resource assistant for new UCSC students. "
            "Your task is to answer the user's question ONLY using the provided context (document snippets). "
            "If the context does not contain the answer, state clearly that the information is not available in the current resources. "
            "Always cite the source document name (e.g., 'Academic_Resource_Center.pdf') if mentioned in the snippet. "
            f"The desired response style is: {style}."
        )

        prompt = (
            f"--- CAMPUS RESOURCE CONTEXT ---\n{context}\n\n"
            f"--- STUDENT QUERY ---\n{query}"
        )
        
        try:
            # 3. Call the Gemini API
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction
                )
            )
            return response.text
            
        except Exception as e:
            return f"❌ Gemini API Error: Failed to generate response. Details: {e}"


# 3. Cached Resource Initialization
@st.cache_resource
def initialize_rag_pipeline():
    """Initializes the Gemini client, Pinecone connection, and the RAG assistant."""
    # Check if required libraries are imported
    if not genai or not Pinecone:
        return None 
        
    try:
        # Retrieve keys and settings securely from secrets.toml
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        pinecone_api_key = st.secrets["PINECONE_API_KEY"]
        pinecone_index_name = st.secrets["PINECONE_INDEX_NAME"] 
        
        # 1. Initialize Gemini Client
        gemini_client = genai.Client(api_key=google_api_key)
        embedding_model = "text-embedding-004" 

        # 2. Initialize Pinecone Client and Index
        pc = Pinecone(api_key=pinecone_api_key)
        if pinecone_index_name not in pc.list_indexes().names:
             raise ValueError(f"Pinecone index '{pinecone_index_name}' does not exist. Please check secrets.toml.")
        pinecone_index = pc.Index(pinecone_index_name)
        
        # 3. Initialize the assistant with the live clients
        assistant = UCSCResourceAssistant(gemini_client, pinecone_index, embedding_model)
        
        return assistant
        
    except KeyError as e:
        st.error(f"Configuration Error: Missing setting in secrets.toml: {e}. Ensure keys for GOOGLE_API_KEY, PINECONE_API_KEY, and PINECONE_INDEX_NAME are present.")
        return None
    except ValueError as e:
        st.error(f"Pinecone Configuration Error: {e}")
        return None
    except Exception as e:
        # Catch all other initialization failures (e.g., general connection errors)
        print(f"Failed to initialize RAG pipeline: {e}") 
        return None


# --- Application Initialization ---

assistant = initialize_rag_pipeline()

# State variables for application flow
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'retrieved_docs' not in st.session_state:
    st.session_state.retrieved_docs = []
if 'generated_answer' not in st.session_state:
    st.session_state.generated_answer = ""
if 'rag_initialized_toast_shown' not in st.session_state:
    st.session_state.rag_initialized_toast_shown = False


# --- Post-Initialization UI Feedback ---
if assistant and not st.session_state.rag_initialized_toast_shown:
    st.toast("✅ RAG Assistant (Gemini/Pinecone) initialized successfully!", icon="🚀")
    st.session_state.rag_initialized_toast_shown = True
    
# --- Custom CSS Styling (for contrast with Ocean Blue theme) ---
st.markdown(
    """
    <style>
    /* Hiding Streamlit's default header */
    .stApp > header {
        visibility: hidden;
    }
    /* Ensuring the main title is visible against the dark background */
    .main-title {
        color: #FFFFFF; /* White for contrast against Ocean Blue background */
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Main Application Logic ---

## 📌 Instructions & Settings

with st.sidebar:
    st.markdown("### 📚 How to Use")
    st.write(
        "Welcome, new Slug! This assistant uses verified campus resources to answer your questions quickly."
    )
    st.write("**Examples:**")
    st.markdown(
        """
        * Where is the Academic Resource Center?
        * What is the cost of a quarterly parking permit?
        * Tell me about the student health insurance plan.
        """
    )
    st.markdown("---")
    
    # Optional Controls
    st.selectbox(
        "Response Style",
        ("Concise Summary", "Detailed Explanation"),
        key="response_style",
        help="Choose how the final answer should be presented."
    )
    st.checkbox(
        "Show Retrieval Details",
        value=True,
        key="show_retrieval_details",
        help="Check this to see the raw document snippets used to generate the answer."
    )


st.markdown('<h1 class="main-title">🌊 UCSC Campus Resource Assistant</h1>', unsafe_allow_html=True)


# Function to run the RAG pipeline
def run_rag_pipeline():
    """Handles the query processing and updates session state."""
    query = st.session_state.query_input.strip()

    if not query:
        st.error("⚠️ Please enter a question to search campus resources.")
        return

    st.session_state.last_query = query
    st.session_state.retrieved_docs = []
    st.session_state.generated_answer = ""

    try:
        with st.spinner(f"🔍 Searching and generating answer for: **{query}**..."):
            
            # 1. Retrieval Step (LIVE Pinecone search)
            retrieved_docs = assistant.retrieve_documents(query)
            st.session_state.retrieved_docs = retrieved_docs
            
            # 2. Generation Step (LIVE Gemini generation)
            generated_answer = assistant.generate_answer(
                query, retrieved_docs, st.session_state.response_style
            )
            st.session_state.generated_answer = generated_answer
            
        st.toast("✅ Answer generated successfully!", icon="✨")
        
    except ConnectionError as e:
        st.error(f"❌ Database/API Error: Could not connect to resource store. Details: {e}")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during processing: {e}")


## 💡 Query Input & Button
col_input, col_button = st.columns([4, 1])

with col_input:
    st.text_input(
        "Ask a Question About UCSC Resources:",
        placeholder="e.g., Where can I get drop-in writing help?",
        key="query_input",
        on_change=lambda: st.session_state.update(generated_answer="", retrieved_docs=[]),
    )

with col_button:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    st.button(
        "Search Campus Resources",
        on_click=run_rag_pipeline,
        type="primary",
        use_container_width=True,
        disabled=not assistant,
    )

st.markdown("---")


## 📃 Output Display Area

if st.session_state.generated_answer:
    
    st.header(f"✨ Answer for: **{st.session_state.last_query}**")
    
    st.write(st.session_state.generated_answer)
    
    st.markdown("---")

    if st.session_state.show_retrieval_details:
        
        if st.session_state.retrieved_docs:
            with st.expander("📂 Sources Used (Retrieval Details)", expanded=False):
                st.info(f"Retrieved **{len(st.session_state.retrieved_docs)}** document snippet(s) to ground the answer.")
                
                # Convert list of dicts to DataFrame for clean display
                df_sources = pd.DataFrame(st.session_state.retrieved_docs)
                df_sources = df_sources.rename(columns={'source': 'Source File/Page', 'snippet': 'Retrieved Text Snippet'})
                
                st.dataframe(df_sources, use_container_width=True)

        else:
            st.warning("No document snippets were retrieved for this query. The answer is based on LLM fallback logic.")

elif not st.session_state.last_query and assistant:
    st.info("👋 Enter your question above and click 'Search Campus Resources' to get started!")

# Final check for initialization failure
if not assistant:
    st.error("❌ The resource assistant failed to initialize. Please check your **secrets.toml** file, the Pinecone index name, and ensure you have installed the `google-genai` and `pinecone` libraries.")