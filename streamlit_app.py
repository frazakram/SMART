#KenshoAI - AI Product fot  Multimodal Analysis & Response
# Enhanced Streamlit Interface for Advanced Document Understanding
import os
import base64
import tempfile
import json
import time
import datetime
from datetime import datetime
from typing import List, Dict
import streamlit as st
from PIL import Image
import pandas as pd
from project2 import AIAgent, ModelFactory
from db import Database

# Initialize database
if "db" not in st.session_state:
    st.session_state.db = Database("KenshoAI_database.db")

# Page configuration
st.set_page_config(
    page_title="🔎 KenshoAI - AI Document Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    border-left: 4px solid #667eea;
}
.user-message {
    background-color: #f0f2f6;
    border-left-color: #667eea;
}
.assistant-message {
    background-color: #e8f4fd;
    border-left-color: #764ba2;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "processed_items" not in st.session_state:
    st.session_state.processed_items = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing_stats" not in st.session_state:
    st.session_state.processing_stats = {
        "total_files": 0,
        "total_items": 0,
        "processing_time": 0,
        "last_processed": None
    }
if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

# Load chat history from database on startup
if not st.session_state.chat_history:
    db_messages = st.session_state.db.get_messages()
    for msg in reversed(db_messages):  # Reverse to maintain chronological order
        metrics = json.loads(msg['metrics']) if msg['metrics'] else None
        st.session_state.chat_history.append({
            "role": msg['role'],
            "content": msg['content'],
            "metrics": metrics
        })

# Load processed items from database on startup
if not st.session_state.processed_items and "agent" in st.session_state and st.session_state.agent:
    db_items = st.session_state.db.get_processed_items()
    for item_dict in db_items:
        # Convert database items to ContentItem objects
        from project2 import ContentItem
        item = ContentItem(
            content_type=item_dict['type'],
            path=item_dict['path'],
            content=item_dict['content'],
            image_data=item_dict['image_base64'],
            metadata=item_dict.get('metadata', {})
        )
        st.session_state.processed_items.append(item)
        if st.session_state.agent:
            st.session_state.agent.items.append(item)

# Header
st.markdown('<h1 class="main-header">🔎 KenshoAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI Tool for  Multimodal Analysis & Response</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload documents, analyze content, and ask questions</p>', unsafe_allow_html=True)

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Select AI Model",
    ["gemini", "anthropic", "openai"],
    index=0
)

api_key = st.sidebar.text_input(
    "API Key",
    value="GEMINI_API_KEY" if model_type == "gemini" else "",
    type="password"
)

model_names = {
    "gemini": ["gemini-2.5-pro", "gemini-1.5-pro"],
    "anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "openai": ["gpt-4o", "gpt-4-turbo", "gpt-4"]
}

model_name = st.sidebar.selectbox(
    "Model Name",
    model_names[model_type]
)
# Add verification model configuration to sidebar
st.sidebar.header("Verification Model Configuration")
use_separate_verification = st.sidebar.checkbox("Use separate model for verification")

verification_model_config = None
if use_separate_verification:
    verification_model_type = st.sidebar.selectbox(
        "Select Verification Model",
        ["gemini", "anthropic", "openai"],
        index=0,
        key="verification_model_type"
    )
    
    verification_api_key = st.sidebar.text_input(
        "Verification API Key",
        value="GEMINI_API_KEY" if verification_model_type == "gemini" else "",
        type="password",
        key="verification_api_key"
    )
    
    verification_model_name = st.sidebar.selectbox(
        "Verification Model Name",
        model_names[verification_model_type],
        key="verification_model_name"
    )
    
    verification_model_config = {
        "type": verification_model_type,
        "api_key": verification_api_key,
        "model_name": verification_model_name
    }

# Database configuration sidebar section
st.sidebar.header("Database Configuration")
db_path = st.sidebar.text_input(
    "Database Path",
    value="KenshoAI_database.db",
    help="Path where the SQLite database will be stored"
)

# Update database if path changes
if db_path != "KenshoAI_database.db" and "db" in st.session_state:
    st.session_state.db = Database(db_path)

# Agent Status Display
if st.session_state.agent_initialized:
    st.sidebar.success("✅ Agent Ready")
    st.sidebar.info(f"Model: {model_type} ({model_name})")
else:
    st.sidebar.warning("⚠️ Agent Not Initialized")

# Initialize Agent Button
if st.sidebar.button("🚀 Initialize Agent", type="primary"):
    if not api_key or api_key in ["GEMINI_API_KEY", "YOUR_API_KEY"]:
        st.sidebar.error("Please enter a valid API key!")
    else:
        with st.spinner("🔄 Initializing AI Agent..."):
            try:
                model_config = {
                    "type": model_type,
                    "api_key": api_key,
                    "model_name": model_name
                }
                st.session_state.agent = AIAgent(model_config, verification_model_config)
                st.session_state.agent_initialized = True
                st.sidebar.success("✅ Agent initialized successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Failed to initialize agent: {str(e)}")
                st.session_state.agent_initialized = False

# Clear Agent Button
if st.session_state.agent_initialized:
    if st.sidebar.button("🗑️ Clear Agent"):
        st.session_state.agent = None
        st.session_state.agent_initialized = False
        st.session_state.processed_items = []
        st.session_state.chat_history = []
        st.session_state.processing_stats = {
            "total_files": 0,
            "total_items": 0,
            "processing_time": 0,
            "last_processed": None
        }
        st.sidebar.success("Agent cleared!")
        st.rerun()


# Dashboard metrics
if st.session_state.processed_items:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📄 Total Items",
            value=len(st.session_state.processed_items),
            delta=f"+{st.session_state.processing_stats['total_items']}" if st.session_state.processing_stats['total_items'] > 0 else None
        )
    
    with col2:
        content_types = {}
        for item in st.session_state.processed_items:
            content_types[item.type] = content_types.get(item.type, 0) + 1
        st.metric(
            label="📊 Content Types",
            value=len(content_types)
        )
    
    with col3:
        st.metric(
            label="💬 Chat Messages",
            value=len(st.session_state.chat_history)
        )
    
    with col4:
        if st.session_state.processing_stats['last_processed']:
            st.metric(
                label="⏰ Last Processed",
                value=st.session_state.processing_stats['last_processed'].strftime("%H:%M")
            )
        else:
            st.metric(label="⏰ Last Processed", value="Never")

# Main content area - enhanced tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Process Content", "💬 Chat Assistant", "📊 Analytics", "⚙️ Settings"])

# Tab 1: Process Content
with tab1:
    if not st.session_state.agent_initialized:
        st.warning("⚠️ Please initialize the AI agent in the sidebar before processing content.")
    else:
        st.header("📤 Content Processing Hub")
        st.markdown("Upload files, enter URLs, or specify local paths to analyze content with AI.")
    
    # Input options
    input_method = st.radio(
        "Select input method:",
        ["Upload Files", "Enter URL", "Enter Local Path"]
    )
    
    if input_method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload documents or images",
            type=["pdf", "jpg", "jpeg", "png", "txt", "xlsx", "xls", "docx", "doc", "csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Process Uploads"):
            with st.spinner("Processing files..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file to temp location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_path = tmp_file.name
                    
                    # Process the file
                    new_items = st.session_state.agent.process_file(temp_path)
                    
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items) 
                        st.session_state.processing_stats['last_processed'] = datetime.now()
                        
                        # Save processed items to database
                        for item in new_items:
                            st.session_state.db.save_processed_item(item.to_dict())
                            
                        st.success(f"Successfully processed {uploaded_file.name}")
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                    
                    # Clean up temp file
                    os.unlink(temp_path)
    
    elif input_method == "Enter URL":
        url = st.text_input("Enter URL to process:")
        if url and st.button("Process URL"):
            if not st.session_state.agent_initialized:
                st.error("⚠️ Please initialize the AI agent in the sidebar before processing content.")
            else:
                with st.spinner("Processing URL..."):
                    new_items = st.session_state.agent.process_url(url)
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)  # Add items to agent
                        st.session_state.processing_stats['last_processed'] = datetime.now()
                        
                        # Save processed items to database
                        for item in new_items:
                            st.session_state.db.save_processed_item(item.to_dict())
                            
                        st.success(f"Processed URL: Found {len(new_items)} content items")
                    else:
                        st.error("Failed to process URL")
    
    elif input_method == "Enter Local Path":
        local_path = st.text_input("Enter local file or directory path:")
        if local_path and st.button("Process Path"):
            if not st.session_state.agent_initialized:
                st.error("⚠️ Please initialize the AI agent in the sidebar before processing content.")
            else:
                with st.spinner("Processing path..."):
                    new_items = st.session_state.agent.process_input(local_path)
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)  # Add items to agent
                        st.session_state.processing_stats['last_processed'] = datetime.now()
                        
                        # Save processed items to database
                        for item in new_items:
                            st.session_state.db.save_processed_item(item.to_dict())
                            
                        st.success(f"Processed path: Found {len(new_items)} content items")
                    else:
                        st.error("Failed to process path")
    
    # Display processed items summary
    if st.session_state.processed_items:
        st.header("Processed Content")
        
        # Summarize content types
        content_types = {}
        for item in st.session_state.processed_items:
            content_types[item.type] = content_types.get(item.type, 0) + 1
        
        # Create summary
        st.subheader("Content Summary")
        for content_type, count in content_types.items():
            st.write(f"- {content_type}: {count} items")
        
        # Special content identified in
        
        # Special content identified in images
                # Special content identified in images
        special_content = []
        for item in st.session_state.processed_items:
            if item.type == "image" and "identified_content" in item.metadata:
                for content_type in item.metadata["identified_content"]:
                    if content_type not in ["image", "photo"]:
                        special_content.append(content_type)
        
        if special_content:
            st.write(f"- Special content identified: {', '.join(set(special_content))}")
        
        # Show a sample of processed items
        with st.expander("View Sample Content"):
            # Display up to 5 items of each type
            displayed_items = 0
            for content_type in ["text", "table", "image", "page"]:
                items = [item for item in st.session_state.processed_items if item.type == content_type][:26]
                if items:
                    st.subheader(f"{content_type.capitalize()} Samples")
                    for item in items:
                        if content_type in ["image", "page"]:
                            try:
                                st.image(item.path, caption=f"Path: {item.path}")
                            except:
                                st.error(f"Could not display image: {item.path}")
                        else:
                            st.text_area(f"Content from {item.path}", item.content, height=150, key=f"item_{displayed_items}_{id(item)}")                            
                        displayed_items += 1
                        if displayed_items >= 26:  # Limit total samples
                            break
                if displayed_items >= 26:
                    break

# Tab 2: Chat with Documents
with tab2:
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.processed_items:
        st.warning("Please process some content first in the 'Process Content' tab.")
    else:
        # Judge metrics selection
        st.subheader("🔍 Judge Metrics Selection")
        available_metrics = ['hallucination', 'relevance', 'completeness', 'accuracy']
        selected_metrics = st.multiselect(
            "Select metrics to evaluate the AI response:",
            options=available_metrics,
            default=['hallucination'],
            help="Choose which metrics to use for evaluating the quality of AI responses"
        )
        
        if not selected_metrics:
            st.warning("Please select at least one metric for evaluation.")
            st.stop()
        
        # Display selected metrics info
        with st.expander("ℹ️ Metric Descriptions"):
            st.write("**Hallucination (0-10):** Measures factual accuracy and groundedness in source material")
            st.write("**Relevance (0-10):** Measures how well the response addresses the specific question asked")
            st.write("**Completeness (0-10):** Measures how comprehensive the response is given available information")
            st.write("**Accuracy (0-10):** Measures the factual correctness of the response against the provided context.")
        
        st.divider()
        
        # Display chat history first
        st.subheader("Conversation")
        for message in st.session_state.chat_history:
            role = message["role"]
            with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
                st.markdown(message["content"])
                
                # Display metrics if available for assistant messages
                if role == "assistant" and "metrics" in message and message["metrics"]:
                    st.divider()
                    st.subheader("Response Quality Metrics")
                    # Dynamically create columns based on the number of metrics
                    num_metrics = len(message["metrics"])
                    cols = st.columns(num_metrics) if num_metrics > 0 else []
                    
                    for i, (metric, data) in enumerate(message["metrics"].items()):
                        with cols[i]:
                            st.markdown(f"<div class='metric-card'><h4>{metric.capitalize()}</h4><h3>{data.get('score', 'N/A')}/10</h3></div>", unsafe_allow_html=True)
                            with st.expander("Justification"):
                                st.write(data.get('justification', 'No justification provided.'))
        
        # Chat input moved to bottom after displaying conversation
        st.divider()
        user_query = st.text_input("Ask a question about your documents:", key="query_input")
        
        if user_query and st.button("Submit Question", key="submit_button"):
            with st.spinner("Generating response..."):
                # Add user query to chat history
                user_message = {"role": "user", "content": user_query}
                st.session_state.chat_history.append(user_message)
                
                # Save user message to database
                st.session_state.db.save_message(
                    role="user",
                    content=user_query,
                    metrics=None
                )
                
                # Get AI response with selected metrics
                response_data = st.session_state.agent.query_content(user_query, metrics=selected_metrics)
                
                # Add assistant response and metrics to chat history
                assistant_message = {
                    "role": "assistant", 
                    "content": response_data.get("response", ""),
                    "metrics": response_data.get("metrics", {})
                }
                st.session_state.chat_history.append(assistant_message)
                
                # Save assistant message to database
                st.session_state.db.save_message(
                    role="assistant",
                    content=response_data.get("response", ""),
                    metrics=response_data.get("metrics", {})
                )
                
                # Rerun to display the new message
                st.rerun()

# Tab 3: Analytics Dashboard
with tab3:
    st.header("📊 Analytics Dashboard")
    
    if not st.session_state.processed_items:
        st.info("📈 Process some content to see analytics and insights.")
    else:
        # Processing statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Processing Statistics")
            stats = st.session_state.processing_stats
            
            metrics_data = {
                "Metric": ["Total Files", "Content Items", "Processing Time (s)", "Avg Items/File"],
                "Value": [
                    str(stats["total_files"]),
                    str(len(st.session_state.processed_items)),
                    f"{stats['processing_time']:.2f}",
                    f"{len(st.session_state.processed_items) / max(stats['total_files'], 1):.1f}"
                ]
            }
            
            # Create DataFrame with explicit string type to avoid Arrow conversion issues
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics['Value'] = df_metrics['Value'].astype(str)
            st.dataframe(df_metrics, use_container_width=True)
        
        with col2:
            st.subheader("💬 Chat Statistics")
            chat_stats = {
                "Metric": ["Total Messages", "User Questions", "AI Responses"],
                "Value": [
                    str(len(st.session_state.chat_history)),
                    str(len([m for m in st.session_state.chat_history if m["role"] == "user"])),
                    str(len([m for m in st.session_state.chat_history if m["role"] == "assistant"]))
                ]
            }
            
            # Create DataFrame with explicit string type to avoid Arrow conversion issues
            df_chat = pd.DataFrame(chat_stats)
            df_chat['Value'] = df_chat['Value'].astype(str)
            st.dataframe(df_chat, use_container_width=True)

        # Add database statistics
        st.subheader("📂 Database Statistics")
        try:
            db_messages = st.session_state.db.get_messages()
            db_items = st.session_state.db.get_processed_items()
            
            db_stats = {
                "Metric": ["Stored Messages", "Stored Content Items", "Database Path"],
                "Value": [
                    str(len(db_messages)),
                    str(len(db_items)),
                    db_path
                ]
            }
            
            df_db = pd.DataFrame(db_stats)
            df_db['Value'] = df_db['Value'].astype(str)
            st.dataframe(df_db, use_container_width=True)
            
            # Show database messages in expandable section
            with st.expander("View Database Messages"):
                if db_messages:
                    message_df = pd.DataFrame([
                        {"ID": msg["id"], "Role": msg["role"], "Content": msg["content"][:100]+"..." if len(msg["content"]) > 100 else msg["content"], "Timestamp": msg["timestamp"]}
                        for msg in db_messages[:50]  # Limit to recent 50
                    ])
                    st.dataframe(message_df, use_container_width=True)
                else:
                    st.info("No messages stored in database yet.")
        except Exception as e:
            st.error(f"Error retrieving database statistics: {str(e)}")

# Tab 4: Settings
with tab4:
    st.header("⚙️ Settings & Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Export Data")
        if st.session_state.processed_items and st.button("📄 Export Content as JSON"):
            # Safely serialize processing stats
            safe_stats = dict(st.session_state.processing_stats)
            if 'last_processed' in safe_stats and safe_stats['last_processed']:
                safe_stats['last_processed'] = safe_stats['last_processed'].isoformat()
            
            export_data = {
                "processed_items_count": len(st.session_state.processed_items),
                "content_types": {item.type: 1 for item in st.session_state.processed_items},
                "stats": safe_stats,
                "export_time": datetime.now().isoformat()
            }
            
            try:
                st.download_button(
                    label="💾 Download",
                    data=json.dumps(export_data, indent=2, default=str),
                    file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
                
        # Add database export option
        if st.button("📤 Export Chat History from Database"):
            try:
                db_messages = st.session_state.db.get_messages()
                if db_messages:
                    export_data = {
                        "messages": db_messages,
                        "export_time": datetime.now().isoformat()
                    }
                    
                    st.download_button(
                        label="💾 Download Chat History",
                        data=json.dumps(export_data, indent=2, default=str),
                        file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No messages to export from database.")
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        st.subheader("🗑️ Clear Data")
        if st.button("🗑️ Clear All Session Data", type="secondary"):
            # Clear all session state data
            st.session_state.processed_items = []
            st.session_state.chat_history = []
            st.session_state.processing_stats = {
                "total_files": 0,
                "total_items": 0,
                "processing_time": 0.0,
                "last_processed": None
            }
            # Also clear agent items if agent exists
            if st.session_state.agent:
                st.session_state.agent.items = []
            
            st.success("✅ All session data cleared successfully!")
            st.rerun()
            
        # Add clear database option
        if st.button("🗑️ Clear Database", type="secondary"):
            try:
                st.session_state.db.clear()
                st.success("✅ Database cleared successfully!")
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")

# Enhanced Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Document Intelligence")
st.sidebar.info(
    "🚀 **Enhanced Features:**\n"
    "• Multi-model AI support\n"
    "• Advanced content analysis\n"
    "• Interactive chat interface\n"
    "• Real-time analytics\n"
    "• Database persistence\n"
    "• Export capabilities"
)

st.sidebar.markdown("---")
st.sidebar.caption("Built by Fraz ©️ 2025")
