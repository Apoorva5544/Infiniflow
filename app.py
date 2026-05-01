import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from rag_engine import (
    process_document, 
    create_vector_store, 
    get_qa_chain, 
    get_vector_store, 
    get_hybrid_retriever,
    rerank_documents,
    list_collections,
    delete_collection
)
import time
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv(override=True)

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="Infiniflow | Analytical Intelligence",
    page_icon="None",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Analytical Aesthetic
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

    /* Global Base */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #1E291B;
    }
    
    .stApp {
        background-color: #F8F9F5; /* Soft Professional Cream */
    }

    /* Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #243010 !important; /* Deep Forest Olive */
        border-right: 1px solid #1A2408;
    }
    
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0rem !important;
        padding-top: 2rem !important;
    }

    .sidebar-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .sidebar-header h1 {
        margin: 0 !important;
        font-size: 1.6rem !important;
        color: #FFFFFF !important;
        letter-spacing: 0.15em !important;
        font-weight: 700 !important;
    }
    
    .sidebar-header p {
        color: #A3B18A !important;
        font-size: 0.75rem !important;
        margin: 0.2rem 0 0 0 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        font-weight: 500;
    }

    /* Sidebar Navigation Buttons */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: transparent !important;
        color: #DAE2CC !important;
        border: 1px solid transparent !important;
        text-align: left !important;
        padding: 0.8rem 1.5rem !important;
        width: 100% !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 0.4rem !important;
    }
    
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: rgba(163, 177, 138, 0.15) !important;
        color: #FFFFFF !important;
        padding-left: 1.8rem !important;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #141B0B !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }

    /* Premium Cards & Metric Blocks */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E4E9DC;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 25px rgba(36, 48, 16, 0.04);
        text-align: left;
    }
    
    .metric-label {
        color: #586B4E;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #243010;
        font-size: 2.2rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        margin-top: 0.2rem;
    }

    .ent-card {
        background: #FFFFFF;
        border: 1px solid #E4E9DC;
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 30px rgba(36, 48, 16, 0.05);
    }
    
    /* Table Rows */
    .kb-row {
        background: white;
        padding: 1rem 1.8rem;
        border: 1px solid #E4E9DC;
        border-radius: 14px;
        margin-bottom: 0.8rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: transform 0.2s, border-color 0.2s;
    }
    .kb-row:hover {
        border-color: #3A5A40;
        transform: translateY(-2px);
    }

    /* Analysis Log Bubbles */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border-radius: 0px !important;
        border: none !important;
        border-bottom: 1px solid #E4E9DC !important;
        padding: 2rem 0rem !important;
        margin-bottom: 0rem !important;
    }
    
    .stChatMessage.assistant { 
        background-color: #FFFFFF !important;
        border-radius: 20px !important;
        border: 1px solid #E4E9DC !important;
        padding: 2rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 2px 15px rgba(0,0,0,0.02) !important;
    }

    /* Custom Primary Button */
    .stButton > button {
        background-color: #3A5A40 !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #243010 !important;
        box-shadow: 0 8px 20px rgba(36, 48, 16, 0.25) !important;
    }

    /* Hide Streamlit elements */
    [data-testid="stHeader"] { background: rgba(255,255,255,0.8); backdrop-filter: blur(10px); }
    div[data-testid="stToolbar"] { visibility: hidden; }
    footer { visibility: hidden; }

    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_collection" not in st.session_state:
    st.session_state.active_collection = None

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.markdown("""
        <div class='sidebar-header'>
            <h1>INFINIFLOW</h1>
            <p>Advanced Analytical Intelligence</p>
        </div>
    """, unsafe_allow_html=True)
    st.divider()
    
    pages = ["Overview", "Repository Central", "Analytical Workspace"]
    
    for p in pages:
        if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
            st.session_state.page = p
            st.rerun()
    
    st.divider()
    if st.session_state.active_collection:
        st.caption("MOUNTED REPOSITORY")
        st.markdown(f"**{st.session_state.active_collection}**")
    
    # Bottom Session Info
    st.markdown("<div style='position: fixed; bottom: 2rem; left: 1rem; color: #D1D9C2; font-size: 0.85rem;'>Precision Engine: v2.4<br>Retrieval: Enabled</div>", unsafe_allow_html=True)

# --- PAGE: OVERVIEW ---
if st.session_state.page == "Overview":
    st.title("Intelligence Dashboard")
    st.write("Real-time metrics and knowledge layer status.")
    
    collections = list_collections()
    
    # Custom Metric Cards for High Contrast
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Knowledge Bases</div>
                <div class='metric-value'>{len(collections)}</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Sync Status</div>
                <div class='metric-value'>Verified</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Consistency</div>
                <div class='metric-value'>99.8%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><h3>Mounted Knowledge Layers</h3>", unsafe_allow_html=True)
    if collections:
        for kb in collections:
            with st.container():
                st.markdown(f"""
                <div class='kb-row'>
                    <span style='font-weight:600; color:#243010; font-size: 1.1rem;'>{kb}</span>
                    <span style='color: #3A5A40; font-size:0.75rem; font-weight:700; background: #E4E9DC; padding: 0.3rem 0.8rem; border-radius: 20px;'>ACTIVE</span>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Begin Analysis on {kb}", key=f"dash_access_{kb}"):
                    st.session_state.active_collection = kb
                    st.session_state.page = "Analytical Workspace"
                    st.rerun()
    else:
        st.info("No knowledge repositories found. Please initialize one in Repository Central.")

# --- PAGE: REPOSITORY CENTRAL ---
elif st.session_state.page == "Repository Central":
    st.title("Data Ingestion & Management")
    st.write("Establish new knowledge layers and ingest documentation.")
    
    # New Library Expansion
    with st.expander("Create New Knowledge Layer", expanded=False):
        new_name = st.text_input("Layer Identifier", placeholder="e.g., Q1_Project_Data")
        if st.button("Establish Layer"):
            if new_name:
                os.makedirs(os.path.join("./chroma_db", new_name), exist_ok=True)
                st.success(f"Storage baseline established for '{new_name}'")
                st.rerun()

    st.divider()
    
    collections = list_collections()
    for kb in collections:
        st.markdown(f"<div class='ent-card'><h3>Layer: {kb}</h3>", unsafe_allow_html=True)
        
        c1, c2 = st.columns([3, 1])
        with c1:
            files = st.file_uploader(f"Upload documentation for ingestion", type=["pdf"], key=f"up_{kb}", accept_multiple_files=True)
            if files and st.button(f"Commence Ingestion for {kb}", key=f"idx_{kb}"):
                with st.spinner("Executing document analysis and vector projection..."):
                    all_chunks = []
                    for f in files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                            t.write(f.getvalue())
                            p = t.name
                        all_chunks.extend(process_document(p))
                        os.remove(p)
                    create_vector_store(all_chunks, kb)
                    st.success(f"Analytical project successfully projected for {kb}")
        
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"Decommission ID: {kb}", key=f"del_{kb}", use_container_width=True):
                delete_collection(kb)
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- PAGE: ANALYTICAL WORKSPACE ---
elif st.session_state.page == "Analytical Workspace":
    if not st.session_state.active_collection:
        st.markdown("<div class='ent-card' style='text-align:center;'><h4>Environment Not Configured</h4><p>Please select a knowledge layer from the Overview to commence analytical work.</p></div>", unsafe_allow_html=True)
        st.stop()
        
    st.title("Intelligence Analysis Terminal")
    st.caption(f"Currently Analyzing: {st.session_state.active_collection}")
    
    # Render History
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if hasattr(msg, "audit"):
                with st.expander("Review Analysis Log"):
                    st.json(msg.audit)

    # Analytical Interaction
    if prompt := st.chat_input(f"Enter query for {st.session_state.active_collection} Analysis..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing insights from analytical project..."):
                vs = get_vector_store(st.session_state.active_collection)
                if vs is None:
                    st.error("Target project not found. Please re-verify in Repository Central.")
                    st.stop()
                
                ret = get_hybrid_retriever(vs)
                chain = get_qa_chain(vs, ret)
                
                start = time.time()
                resp = chain.invoke({"input": prompt, "chat_history": st.session_state.chat_history})
                total_time = round(time.time() - start, 2)
                
                st.markdown(resp["answer"])
                
                audit = {"latency": f"{total_time}s", "projection_hits": len(resp["context"]), "engine": "Analytical Model V3"}
                with st.expander("Review Analysis Log"):
                    st.json(audit)
                
                # Source List
                citations = list(set([doc.metadata.get("source", "Unknown") for doc in resp["context"]]))
                st.markdown("---")
                st.markdown("**DOCUMENT REFERENCES:**")
                for c in citations: st.markdown(f"• `{c}`")

                # Persistence
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                ai_msg = AIMessage(content=resp["answer"])
                ai_msg.audit = audit
                st.session_state.chat_history.append(ai_msg)
