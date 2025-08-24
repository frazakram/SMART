# streamlit_app.py
import os
import base64
import tempfile
import streamlit as st
from PIL import Image
from project2 import AIAgent, ModelFactory

# Page configuration
st.set_page_config(
    page_title="Document Understanding AI",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "processed_items" not in st.session_state:
    st.session_state.processed_items = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Title and description
st.title("Document Understanding AI")
st.markdown("Upload documents, images, or provide URLs to analyze and query content.")

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Select AI Model",
    ["gemini", "anthropic", "openai"],
    index=0
)

api_key = st.sidebar.text_input(
    "API Key",
    value="AIzaSyA03TDmMCySeHgisstcCLeBurY9NnyCytE" if model_type == "gemini" else "",
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
        value="AIzaSyA03TDmMCySeHgisstcCLeBurY9NnyCytE" if verification_model_type == "gemini" else "",
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
# Update streamlit_app.py to add the metric selection option

# Add the metric selection to the sidebar
st.sidebar.header("Evaluation Metrics")
evaluation_metric = st.sidebar.radio(
    "Select evaluation metric",
    ["hallucination", "faithfulness"],
    help="Hallucination: Detects content not supported by sources. Faithfulness: Measures how well response represents source materials."
)

# Update the agent initialization to include the selected metric
if st.sidebar.button("Initialize Agent") or st.session_state.agent is None:
    with st.spinner("Initializing AI Agent..."):
        model_config = {
            "type": model_type,
            "api_key": api_key,
            "model_name": model_name
        }
        st.session_state.agent = AIAgent(model_config, verification_model_config, evaluation_metric)
        st.sidebar.success("Agent initialized!")

# Update the agent initialization


# Initialize agent


# Main content area - tabs for different functions
tab1, tab2 = st.tabs(["Process Content", "Chat with Documents"])

# Tab 1: Process Content
with tab1:
    st.header("Upload or Link Content")
    
    # Input options
    input_method = st.radio(
        "Select input method:",
        ["Upload Files", "Enter URL", "Enter Local Path"]
    )
    
    if input_method == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload documents or images",
            type=["pdf", "jpg", "jpeg", "png", "txt"],
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
                    # In the streamlit_app.py file, update these sections:

# For the "Upload Files" section (around line 94):
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)  # Add items to agent
                        st.success(f"Processed {uploaded_file.name}: Found {len(new_items)} content items")

                    # For the "Enter URL" section (around line 108):
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)  # Add items to agent
                        st.success(f"Processed URL: Found {len(new_items)} content items")

                    # For the "Enter Local Path" section (around line 119):
                    if new_items:
                        st.session_state.processed_items.extend(new_items)
                        st.session_state.agent.items.extend(new_items)  # Add items to agent
                        st.success(f"Processed path: Found {len(new_items)} content items")
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                    
                    # Clean up temp file
                    os.unlink(temp_path)
    
    elif input_method == "Enter URL":
        url = st.text_input("Enter URL to process:")
        if url and st.button("Process URL"):
            with st.spinner("Processing URL..."):
                new_items = st.session_state.agent.process_url(url)
                if new_items:
                    st.session_state.processed_items.extend(new_items)
                    st.success(f"Processed URL: Found {len(new_items)} content items")
                else:
                    st.error("Failed to process URL")
    
    elif input_method == "Enter Local Path":
        local_path = st.text_input("Enter local file or directory path:")
        if local_path and st.button("Process Path"):
            with st.spinner("Processing path..."):
                new_items = st.session_state.agent.process_input(local_path)
                if new_items:
                    st.session_state.processed_items.extend(new_items)
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
                items = [item for item in st.session_state.processed_items if item.type == content_type][:5]
                if items:
                    st.subheader(f"{content_type.capitalize()} Samples")
                    for item in items:
                        if content_type in ["image", "page"]:
                            try:
                                st.image(item.path, caption=f"Path: {item.path}")
                            except:
                                st.error(f"Could not display image: {item.path}")
                        else:
                            # st.text_area(f"Content from {item.path}", item.content, height=150)
                            st.text_area(f"Content from {item.path}", item.content, height=150, key=f"item_{displayed_items}_{id(item)}")                            
                        displayed_items += 1
                        if displayed_items >= 10:  # Limit total samples
                            break
                if displayed_items >= 10:
                    break

# Tab 2: Chat with Documents
with tab2:
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.processed_items:
        st.warning("Please process some content first in the 'Process Content' tab.")
    else:
        # Chat input
        user_query = st.text_input("Ask a question about your documents:")
        
        if user_query and st.button("Submit Question"):
            with st.spinner("Generating response..."):
                # Add user query to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                
                # Get AI response
                response = st.session_state.agent.query_content(user_query)
                
                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        st.subheader("Conversation")
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
            st.divider()

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This application uses AI models to analyze documents, extract content, "
    "and answer questions about the content."
)