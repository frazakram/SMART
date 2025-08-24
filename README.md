# Document Understanding AI Chatbot

A multimodal AI agent that can analyze various document types (PDFs, images, text), extract content, and answer questions about them.

## Features

- **Multiple AI Model Support**: Works with OpenAI, Anthropic, and Google's Gemini models
- **Document Processing**: Handles PDFs, images, and text files
- **Content Extraction**: Identifies and extracts text, tables, and visual elements
- **Multimodal Understanding**: Can analyze and answer questions about visual content
- **Evaluation Metrics**: Supports hallucination and faithfulness evaluation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/document-understanding-ai.git
cd document-understanding-ai

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the root directory with your API keys:
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key

## Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Deployment

This application can be deployed on Render:

1. Fork this repository to your GitHub account
2. Create a new Web Service on Render
3. Connect your GitHub repository
4. Configure the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
5. Add your API keys as environment variables

## Project Structure

- `streamlit_app.py`: Main application file with Streamlit UI
- `project2.py`: Core functionality and model abstractions
- `requirements.txt`: Project dependencies
- `SMART/`: Additional utility functions and components

## License

MIT
