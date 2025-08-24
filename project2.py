# Multimodal AI Agent for Document Understanding
# Identifies content in images and allows querying them

import argparse
import base64
import glob
import io
import json
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import google.generativeai as genai
import pandas as pd
import pymupdf
import requests
from PIL import Image
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Suppress warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Model Abstraction Framework

class AIModel(ABC):
    """Abstract base class for AI models"""
    
    @abstractmethod
    def generate_content(self, parts: List[Dict]) -> str:
        """Generate content from the model"""
        pass
    
    @abstractmethod
    def identify_image_content(self, image: Image.Image, encoded_image: str) -> List[str]:
        """Identify content in an image"""
        pass

class GeminiModel(AIModel):
    """Implementation for Google's Gemini model"""
    
    def __init__(self, api_key: str, model_name: str = 'gemini-2.5-pro'):
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate_content(self, parts: List[Dict]) -> str:
        """Generate content using Gemini"""
        try:
            response = self.model.generate_content(parts)
            return response.text
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"Error generating response: {str(e)}"
    
    def identify_image_content(self, image: Image.Image, encoded_image: str) -> List[str]:
        """Identify content in an image using Gemini"""
        try:
            parts = [
                {"text": "Identify what type of content this image contains. Pay special attention to graphs, charts, bar graphs, and histograms. Respond ONLY with relevant tags from this list: image, photo, text, document, handwriting, graph, chart, bar graph, histogram, diagram, table, infographic, map, drawing, screenshot. Multiple tags are allowed."},
                {"inline_data": {
                    "mime_type": "image/png",
                    "data": encoded_image
                }}
            ]
            
            response = self.model.generate_content(parts)
            
            # Extract content types
            content_types = []
            for tag in ["image", "photo", "text", "document", "handwriting", 
                        "graph", "chart", "diagram", "table", "infographic", 
                        "map", "drawing", "screenshot"]:
                if tag.lower() in response.text.lower():
                    content_types.append(tag)
            
            # Always include at least "image" as a fallback
            if not content_types:
                content_types = ["image"]
                
            return content_types
            
        except Exception as e:
            logger.error(f"Error identifying content: {e}")
            return ["image"]  # Default to image

class AnthropicModel(AIModel):
    """Implementation for Anthropic's Claude model"""
    
    def __init__(self, api_key: str, model_name: str = 'claude-3-opus-20240229'):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            logger.error("Anthropic SDK not installed. Run 'pip install anthropic'")
            raise
    
    def generate_content(self, parts: List[Dict]) -> str:
        """Generate content using Claude"""
        try:
            # Convert parts format from Gemini to Claude format
            system_prompt = ""
            user_message = ""
            images = []
            
            for part in parts:
                if 'text' in part:
                    if 'You are a helpful AI agent' in part['text']:  # System prompt
                        system_prompt = part['text']
                    else:
                        user_message += part['text'] + "\n\n"
                elif 'inline_data' in part and part['inline_data']['mime_type'].startswith('image/'):
                    # Add image to the message
                    image_data = part['inline_data']['data']
                    images.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}})
            
            # Build the message content
            content = []
            
            # Add text
            if user_message:
                content.append({"type": "text", "text": user_message})
            
            # Add images
            for image in images:
                content.append(image)
            
            # Make API call
            response = self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[{"role": "user", "content": content}],
                max_tokens=4096
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            return f"Error generating response: {str(e)}"
    
    def identify_image_content(self, image: Image.Image, encoded_image: str) -> List[str]:
        """Identify content in an image using Claude"""
        try:
            content = [
                {"type": "text", "text": "Identify what type of content this image contains. Respond ONLY with relevant tags from this list: image, photo, text, document, handwriting, graph, chart, diagram, table, infographic, map, drawing, screenshot. Multiple tags are allowed."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded_image}}
            ]
            
            response = self.client.messages.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=100
            )
            
            # Extract content types
            response_text = response.content[0].text
            content_types = []
            for tag in ["image", "photo", "text", "document", "handwriting", 
                        "graph", "chart", "diagram", "table", "infographic", 
                        "map", "drawing", "screenshot"]:
                if tag.lower() in response_text.lower():
                    content_types.append(tag)
            
            # Always include at least "image" as a fallback
            if not content_types:
                content_types = ["image"]
                
            return content_types
            
        except Exception as e:
            logger.error(f"Error identifying content with Claude: {e}")
            return ["image"]  # Default to image

class OpenAIModel(AIModel):
    """Implementation for OpenAI's models"""
    
    def __init__(self, api_key: str, model_name: str = 'gpt-4o'):
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model_name = model_name
        except ImportError:
            logger.error("OpenAI SDK not installed. Run 'pip install openai'")
            raise
    
    def generate_content(self, parts: List[Dict]) -> str:
        """Generate content using OpenAI"""
        try:
            # Convert parts format from Gemini to OpenAI format
            system_content = ""
            user_content = []
            
            for part in parts:
                if 'text' in part:
                    if 'You are a helpful AI agent' in part['text']:  # System prompt
                        system_content = part['text']
                    else:
                        user_content.append({"type": "text", "text": part['text']})
                elif 'inline_data' in part and part['inline_data']['mime_type'].startswith('image/'):
                    # Add image to the message
                    image_data = part['inline_data']['data']
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    })
            
            # Make API call
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error generating response: {str(e)}"
    
    def identify_image_content(self, image: Image.Image, encoded_image: str) -> List[str]:
        """Identify content in an image using OpenAI"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify what type of content this image contains. Respond ONLY with relevant tags from this list: image, photo, text, document, handwriting, graph, chart, diagram, table, infographic, map, drawing, screenshot. Multiple tags are allowed."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100
            )
            
            # Extract content types
            response_text = response.choices[0].message.content
            content_types = []
            for tag in ["image", "photo", "text", "document", "handwriting", 
                        "graph", "chart", "diagram", "table", "infographic", 
                        "map", "drawing", "screenshot"]:
                if tag.lower() in response_text.lower():
                    content_types.append(tag)
            
            # Always include at least "image" as a fallback
            if not content_types:
                content_types = ["image"]
                
            return content_types
            
        except Exception as e:
            logger.error(f"Error identifying content with OpenAI: {e}")
            return ["image"]  # Default to image

class ModelFactory:
    """Factory class to create AI models"""
    
    @staticmethod
    def create_model(model_type: str, api_key: str, model_name: str = None) -> AIModel:
        """Create and return an AI model based on the specified type"""
        model_type = model_type.lower()
        
        if model_type == "gemini":
            default_model = "gemini-2.5-pro"
            return GeminiModel(api_key, model_name or default_model)
        elif model_type == "anthropic":
            default_model = "claude-3-opus-20240229"
            return AnthropicModel(api_key, model_name or default_model)
        elif model_type == "openai":
            default_model = "gpt-4o"
            return OpenAIModel(api_key, model_name or default_model)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class ContentItem:
    """Class to represent an identified content item"""
    def __init__(self, 
                 content_type: str, 
                 path: str, 
                 content: Union[str, bytes], 
                 image_data: Optional[str] = None,
                 metadata: Optional[Dict] = None):
        self.type = content_type  # 'text', 'image', 'table', 'graph', 'chart'
        self.path = path
        self.content = content
        self.image_data = image_data  # Base64 encoded image
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API calls"""
        return {
            "type": self.type,
            "path": self.path,
            "content": self.content if isinstance(self.content, str) else None,
            "image": self.image_data,
            "metadata": self.metadata
        }

class AIAgent:
    """Multimodal AI Agent for content understanding and Q&A"""
    
    def __init__(self, model_config: Dict = None, verification_model_config: Dict = None, evaluation_metric: str = 'hallucination'):
        self.items = []  # List of identified content items
        self.base_dir = "agent_data"
        self.create_directories()
        self.evaluation_metric = evaluation_metric  # 'hallucination' or 'faithfulness'
        
        # Configure the primary AI model
        if model_config is None:
            # Default to Gemini if no configuration provided
            model_config = {
                "type": "gemini",
                "api_key": "GEMINI_API_KEY",  # Replace with your key
                "model_name": "gemini-2.5-pro"
            }
        
        # Create the primary AI model
        self.model = ModelFactory.create_model(
            model_type=model_config["type"],
            api_key=model_config["api_key"],
            model_name=model_config.get("model_name")
        )
        
        # Configure the verification model (if provided, otherwise use the primary model)
        if verification_model_config:
            self.verification_model = ModelFactory.create_model(
                model_type=verification_model_config["type"],
                api_key=verification_model_config["api_key"],
                model_name=verification_model_config.get("model_name")
            )
        else:
            self.verification_model = self.model
        
        # Content identifier uses the primary model
        self.content_identifier = ContentIdentifier(self.model)

    
    
    

    
    def create_directories(self):
        """Create necessary directories for storing processed content"""
        directories = ["images", "text", "tables", "processed", "extracted"]
        for dir_name in directories:
            os.makedirs(os.path.join(self.base_dir, dir_name), exist_ok=True)
    
    def process_input(self, input_path: str) -> List[ContentItem]:
        """Process the input (file path, URL, or directory)"""
        logger.info(f"Processing input: {input_path}")
        
        if input_path.startswith(('http://', 'https://')):
            return self.process_url(input_path)
        elif os.path.isdir(input_path):
            return self.process_directory(input_path)
        elif os.path.isfile(input_path):
            return self.process_file(input_path)
        else:
            logger.error(f"Invalid input path: {input_path}")
            return []
    
    def process_url(self, url: str) -> List[ContentItem]:
        """Process a URL (image or document)"""
        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                logger.error(f"Failed to download from URL: {url}, Status: {response.status_code}")
                return []
            
            # Determine file type from headers or URL
            content_type = response.headers.get('Content-Type', '')
            filename = url.split('/')[-1].split('?')[0]  # Extract filename from URL
            
            if not filename:
                if 'image' in content_type:
                    extension = content_type.split('/')[-1]
                    filename = f"downloaded_image.{extension}"
                elif 'pdf' in content_type:
                    filename = "downloaded_document.pdf"
                else:
                    filename = "downloaded_file"
            
            # Save the file
            filepath = os.path.join(self.base_dir, "processed", filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded file to: {filepath}")
            return self.process_file(filepath)
            
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return []
    
    def process_directory(self, directory: str) -> List[ContentItem]:
        """Process all files in a directory"""
        items = []
        
        # Get all supported files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
        document_extensions = ['*.pdf', '*.docx', '*.txt']
        
        all_files = []
        for ext in image_extensions + document_extensions:
            all_files.extend(glob.glob(os.path.join(directory, ext)))
            all_files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        logger.info(f"Found {len(all_files)} files in directory {directory}")
        
        for file_path in tqdm(all_files, desc="Processing files"):
            items.extend(self.process_file(file_path))
        
        return items
    
    def process_file(self, file_path: str) -> List[ContentItem]:
        """Process a single file based on its type"""
        items = []
        
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Process based on file type
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
                items.extend(self.process_image_file(file_path))
            elif file_ext == '.pdf':
                items.extend(self.process_pdf_file(file_path))
            elif file_ext in ['.txt', '.csv']:
                items.extend(self.process_text_file(file_path))
            else:
                logger.warning(f"Unsupported file type: {file_ext}")
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        
        return items
    
    def process_image_file(self, image_path: str) -> List[ContentItem]:
        """Process a single image file"""
        items = []
        
        try:
            # Get the base filename
            image_name = os.path.basename(image_path)
            
            # Process the image
            processed_path = os.path.join(self.base_dir, "processed", image_name)
            
            # Load the image using PIL
            with Image.open(image_path) as img:
                # Save as PNG for consistency
                if not image_path.lower().endswith('.png'):
                    processed_path = os.path.splitext(processed_path)[0] + '.png'
                    img.save(processed_path, 'PNG')
                else:
                    # If already PNG, just copy
                    img.save(processed_path)
            
            # Base64 encode the image for API calls
            with open(processed_path, 'rb') as f:
                encoded_image = base64.b64encode(f.read()).decode('utf8')
            
            # Identify content in the image
            content_types = self.content_identifier.identify_image_content(img, encoded_image)
            
            # Create a content item
            item = ContentItem(
                content_type="image",
                path=processed_path,
                content=image_name,
                image_data=encoded_image,
                metadata={"identified_content": content_types}
            )
            
            items.append(item)
            logger.info(f"Processed image: {image_name} - Identified: {content_types}")
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
        
        return items
    
    def process_pdf_file(self, pdf_path: str) -> List[ContentItem]:
        """Process a PDF file and extract content"""
        items = []
        
        try:
            # Open the PDF document
            doc = pymupdf.open(pdf_path)
            filename = os.path.basename(pdf_path)
            num_pages = len(doc)
            
            logger.info(f"Processing PDF: {filename} ({num_pages} pages)")
            
            # Create text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700, 
                chunk_overlap=200, 
                length_function=len
            )
            
            # Process each page
            for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    # Process text chunks
                    chunks = text_splitter.split_text(text)
                    for i, chunk in enumerate(chunks):
                        text_file_name = f"{self.base_dir}/text/{filename}_text_{page_num}_{i}.txt"
                        with open(text_file_name, 'w', encoding='utf-8') as f:
                            f.write(chunk)
                        
                        item = ContentItem(
                            content_type="text",
                            path=text_file_name,
                            content=chunk,
                            metadata={"page": page_num, "chunk": i}
                        )
                        items.append(item)
                
                # Extract tables
                try:
                    tables = page.find_tables()
                    for i, table in enumerate(tables):
                        # Extract table data
                        table_data = table.extract()
                        
                        # Save as CSV
                        table_file_name = f"{self.base_dir}/tables/{filename}_table_{page_num}_{i}.csv"
                        
                        # Convert table data to DataFrame and save
                        df = pd.DataFrame(table_data)
                        df.to_csv(table_file_name, index=False)
                        
                        item = ContentItem(
                            content_type="table",
                            path=table_file_name,
                            content=df.to_string(),
                            metadata={"page": page_num}
                        )
                        items.append(item)
                except Exception as e:
                    logger.error(f"Error extracting tables from page {page_num}: {e}")
                
                # Extract images
                images = page.get_images()
                for idx, image in enumerate(images):
                    try:
                        xref = image[0]
                        # Get image data using PyMuPDF
                        pix = pymupdf.Pixmap(doc, xref)
                        
                        # Convert to RGB if needed
                        if pix.colorspace and pix.colorspace.n > 3:
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                        elif pix.colorspace is None:
                            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                            
                        image_name = f"{self.base_dir}/images/{filename}_image_{page_num}_{idx}_{xref}.png"
                        pix.save(image_name)
                        
                        # Verify the file exists and has content
                        if os.path.exists(image_name) and os.path.getsize(image_name) > 0:
                            with open(image_name, 'rb') as f:
                                encoded_image = base64.b64encode(f.read()).decode('utf8')
                            
                            # Load with PIL to analyze content
                            img = Image.open(image_name)
                            content_types = self.content_identifier.identify_image_content(img, encoded_image)
                            
                            item = ContentItem(
                                content_type="image",
                                path=image_name,
                                content=f"Image from page {page_num}",
                                image_data=encoded_image,
                                metadata={
                                    "page": page_num,
                                    "identified_content": content_types
                                }
                            )
                            items.append(item)
                    except Exception as e:
                        logger.error(f"Error processing image {idx} on page {page_num}: {e}")
                
                # Save page image
                pix = page.get_pixmap()
                page_path = os.path.join(self.base_dir, f"images/page_{filename}_{page_num:03d}.png")
                pix.save(page_path)
                with open(page_path, 'rb') as f:
                    page_image = base64.b64encode(f.read()).decode('utf8')
                
                item = ContentItem(
                    content_type="page",
                    path=page_path,
                    content=f"Full page {page_num}",
                    image_data=page_image,
                    metadata={"page": page_num}
                )
                items.append(item)
        
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
        
        return items
    
    def process_text_file(self, text_path: str) -> List[ContentItem]:
        """Process a text file"""
        items = []
        
        try:
            filename = os.path.basename(text_path)
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700, 
                chunk_overlap=200, 
                length_function=len
            )
            
            # Process text chunks
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                text_file_name = f"{self.base_dir}/text/{filename}_chunk_{i}.txt"
                with open(text_file_name, 'w', encoding='utf-8') as f:
                    f.write(chunk)
                
                item = ContentItem(
                    content_type="text",
                    path=text_file_name,
                    content=chunk,
                    metadata={"chunk": i}
                )
                items.append(item)
        
        except Exception as e:
            logger.error(f"Error processing text file {text_path}: {e}")
        
        return items
    
    def query_content(self, query: str, max_items: int = 10) -> str:
        """Query the identified content items"""
        if not self.items:
            return "No content has been processed yet. Please provide a file, URL, or directory to analyze."
        
        # Select relevant items based on content type
        text_items = [item for item in self.items if item.type == "text"]
        table_items = [item for item in self.items if item.type == "table"]
        image_items = [item for item in self.items if item.type == "image"]
        page_items = [item for item in self.items if item.type == "page"]
        graph_items = [item for item in self.items if "graph" in item.metadata.get("identified_content", [])]
        chart_items = [item for item in self.items if "chart" in item.metadata.get("identified_content", [])]
       # In query_content method (around line 670), add this specific detection:
        visualization_items = [item for item in self.items if any(vis_type in item.metadata.get("identified_content", []) 
                            for vis_type in ["graph", "chart", "bar graph", "histogram"])]

# And then in the query keywords section (around line 690):

        # Select items based on the query keywords
        selected_items = []
        

        # Keywords to help with content selection
        if any(keyword in query.lower() for keyword in ["image", "picture", "photo", "show"]):
            selected_items.extend(image_items[:11])
        
        if any(keyword in query.lower() for keyword in ["text", "say", "write", "content"]):
            if text_items:
                # Include more text items from different pages
                pages_covered = set()
                for item in text_items:
                    page_num = item.metadata.get("page")
                    if page_num is not None and page_num not in pages_covered and len(pages_covered) < 5:
                        selected_items.append(item)
                        pages_covered.add(page_num)
        
        if any(keyword in query.lower() for keyword in ["table", "data", "column", "row"]):
            selected_items.extend(table_items[:11])
        
        if any(keyword in query.lower() for keyword in ["graph", "plot", "trend", "bar graph", "bar chart", "histogram","box plot","heat map","pie chart"]):
            selected_items.extend(graph_items[:11])
        
        if any(keyword in query.lower() for keyword in ["chart", "diagram", "visualization"]):
            selected_items.extend(chart_items[:11])
        
        if any(keyword in query.lower() for keyword in ["page", "whole", "entire", "complete", "all"]):
            if page_items:
                # Include more pages - up to 5
                selected_items.extend(page_items[:11])
        if any(keyword in query.lower() for keyword in ["graph", "plot", "trend", "bar graph", "bar chart", "histogram","box plot","heat map","pie chart"]):
                selected_items.extend(visualization_items[:11])
        
        # If no specific content type was identified in the query, use a balanced approach
        if not selected_items:
            if text_items:
                # Include text from different pages
                pages_covered = set()
                for item in text_items:
                    page_num = item.metadata.get("page")
                    if page_num is not None and page_num not in pages_covered and len(pages_covered) < 5:
                        selected_items.append(item)
                        pages_covered.add(page_num)
            
            if table_items:
                selected_items.extend(table_items[:11])
            
            if image_items:
                selected_items.extend(image_items[:11])
            
            # Always include a few pages if nothing else was selected
            if not selected_items and page_items:
                selected_items.extend(page_items[:11])
        
        # Limit to max_items
        selected_items = selected_items[:max_items]
        
        if not selected_items:
            return "I couldn't find relevant content to answer your question. Please try a different question or provide more content to analyze."
        
        # Prepare items for the model
        items_for_model = [item.to_dict() for item in selected_items]
        return self.query_model(query, items_for_model)
    def verify_response(self, response: str, content_items: List[Dict]) -> Dict:
        """Check if the response is grounded in the provided content, including images"""
    # Prepare verification prompt
        verification_parts = []
        
        # Add source materials including images
        has_images = False
        for item in content_items:
            if item['type'] == 'text' or item['type'] == 'table':
                verification_parts.append({"text": item['content']})
            elif item['type'] == 'image' and item.get('image'):
                # Include images for verification
                verification_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": item['image']
                    }
                })
                has_images = True
        
        # Create a verification prompt appropriate for the content type
        verification_text = f"""
    RESPONSE TO VERIFY: {response}

    VERIFICATION INSTRUCTIONS:
    """
        
        if has_images:
            verification_text += """
    For visual content:
    1. Verify if the descriptions of people, objects, colors, actions, and scenes match what's visible in the images
    2. Be lenient about subjective interpretations (e.g., describing a color as "light blue" vs "cyan")
    3. Focus on factual accuracy, not completeness (missing details are not hallucinations)
    """
        
        verification_text += """
    For all content:
    1. Identify specific statements that aren't supported by the source materials
    2. Calculate a hallucination score from 0-10 where:
    - 0-2: Minimal or no hallucination (factual, well-grounded)
    - 3-5: Minor hallucinations (small inaccuracies but mostly correct)
    - 6-8: Significant hallucinations (major details are wrong)
    - 9-10: Severe hallucinations (completely fabricated)
    3. Provide a brief explanation for your score
    4. Return your analysis in JSON format with these fields:
    - hallucination_score: number from 0-10
    - explanation: string explaining the rationale for the score
    - ungrounded_claims: list of statements not supported by sources
    - confidence: your confidence in this assessment (0-1)
    """
        
        verification_parts.append({"text": verification_text})
        
        # Get verification result using the verification model
        verification_result = self.verification_model.generate_content(verification_parts)
        
        # Parse JSON from response
        try:
            # Extract JSON if embedded in text
            json_text = verification_result
            if "{" in verification_result and "}" in verification_result:
                json_text = verification_result[verification_result.find("{"):verification_result.rfind("}")+1]
            return json.loads(json_text)
        except:
            # Fallback if JSON parsing fails
            return {
                "hallucination_score": 5,
                "explanation": "Failed to parse verification result",
                "ungrounded_claims": ["Verification system error"],
                "confidence": 0
            }
    def evaluate_faithfulness(self, response: str, content_items: List[Dict]) -> Dict:
        """Evaluate the faithfulness of the response to the provided content"""
        # Prepare verification parts
        verification_parts = []
        
        # Add source materials including images
        has_images = False
        for item in content_items:
            if item['type'] == 'text' or item['type'] == 'table':
                verification_parts.append({"text": item['content']})
            elif item['type'] == 'image' and item.get('image'):
                # Include images for verification
                verification_parts.append({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": item['image']
                    }
                })
                has_images = True
        
        # Create a verification prompt for faithfulness
        verification_text = f"""
    RESPONSE TO EVALUATE: {response}

    FAITHFULNESS EVALUATION INSTRUCTIONS:
    """
        
        if has_images:
            verification_text += """
    For visual content:
    1. Evaluate how well the response is grounded in the visual information provided
    2. Check if claims about visual elements are supported by what's visible in the images
    3. Determine if the response is faithful to the content and meaning conveyed in the images
    """
        
        verification_text += """
    For all content:
    1. Evaluate the faithfulness of the response to the source materials
    2. Calculate a faithfulness score from 0-10 where:
    - 0-2: Not faithful at all (contradicts or ignores source content)
    - 3-5: Low faithfulness (misrepresents or distorts source content)
    - 6-8: Moderately faithful (mostly represents source content accurately)
    - 9-10: Highly faithful (accurately represents source content)
    3. Provide a brief explanation for your score
    4. Return your analysis in JSON format with these fields:
    - faithfulness_score: number from 0-10
    - explanation: string explaining the rationale for the score
    - unfaithful_claims: list of statements that misrepresent the source content
    - confidence: your confidence in this assessment (0-1)
    """
        
        verification_parts.append({"text": verification_text})
        
        # Get verification result using the verification model
        verification_result = self.verification_model.generate_content(verification_parts)
        
        # Parse JSON from response
        try:
            # Extract JSON if embedded in text
            json_text = verification_result
            if "{" in verification_result and "}" in verification_result:
                json_text = verification_result[verification_result.find("{"):verification_result.rfind("}")+1]
            return json.loads(json_text)
        except:
            # Fallback if JSON parsing fails
            return {
                "faithfulness_score": 5,
                "explanation": "Failed to parse faithfulness evaluation result",
                "unfaithful_claims": ["Evaluation system error"],
                "confidence": 0
            }
        
    def query_model(self, prompt: str, content_items: List[Dict]) -> str:
        """Query the AI model with text and images"""
        # Prepare content parts for the model
        parts = []
        
        # System instruction as text
        parts.append({
            "text": "You are a helpful AI agent for content understanding and question answering. "
                "The provided text, tables, and images are relevant information retrieved to help answer the question "
                "with accuracy, completeness, clarity, and relevance. Analyze all content carefully."
        })
        
        # Add content from items
        for item in content_items:
            if item['type'] == 'text' or item['type'] == 'table':
                parts.append({"text": item['content']})
            else:
                # For images, use the base64 content
                try:
                    parts.append({"inline_data": {
                        "mime_type": "image/png",
                        "data": item['image']
                    }})
                except Exception as e:
                    logger.error(f"Error processing image for model: {e}")
        
        # Add the user prompt
        parts.append({"text": prompt})
        response = self.model.generate_content(parts)

        # Evaluate response based on selected metric
        if self.evaluation_metric == 'hallucination':
            verification = self.verify_response(response, content_items)
            if verification.get("hallucination_score", 0) > 5:
                response = f"[⚠️ Hallucination Warning: Score {verification.get('hallucination_score')}/10] \n" + \
                        f"Reason: {verification.get('explanation', 'High risk of inaccurate information')} \n\n{response}"
            else:
                response = f"[✓ Factual Content: Score {verification.get('hallucination_score')}/10] \n" + \
                        f"Note: {verification.get('explanation', 'Information appears accurate')} \n\n{response}"
        else:  # faithfulness
            verification = self.evaluate_faithfulness(response, content_items)
            if verification.get("faithfulness_score", 0) < 6:
                response = f"[⚠️ Low Faithfulness Warning: Score {verification.get('faithfulness_score')}/10] \n" + \
                        f"Reason: {verification.get('explanation', 'Response may not accurately represent the source materials')} \n\n{response}"
            else:
                response = f"[✓ Faithful Content: Score {verification.get('faithfulness_score')}/10] \n" + \
                        f"Note: {verification.get('explanation', 'Response accurately represents the source materials')} \n\n{response}"
        
        return response
            # Call model API
            # return self.model.generate_content(parts)
    
    def run(self):
        """Run the agent interactively"""
        print("=" * 80)
        print("  Multimodal AI Agent for Content Understanding")
        print("  Provide a file path, URL, or directory to analyze content")
        print("=" * 80)
        
        while True:
            user_input = input("\nEnter path/URL to analyze or a question (type 'quit' to exit): ")
            
            if user_input.lower() == 'quit':
                print("Exiting the AI agent. Goodbye!")
                break
            
            # Check if this is a path/URL or a question
            if (user_input.startswith(('http://', 'https://')) or 
                os.path.exists(user_input) or
                any(ext in user_input.lower() for ext in ['.jpg', '.png', '.pdf', '/'])):
                
                # This is a path/URL to process
                print(f"Processing content from: {user_input}")
                new_items = self.process_input(user_input)
                
                if new_items:
                    self.items.extend(new_items)
                    content_types = {}
                    for item in new_items:
                        content_types[item.type] = content_types.get(item.type, 0) + 1
                    
                    print(f"\nIdentified content:")
                    for content_type, count in content_types.items():
                        print(f"- {content_type}: {count} items")
                    
                    special_content = []
                    for item in new_items:
                        if item.type == "image" and "identified_content" in item.metadata:
                            for content_type in item.metadata["identified_content"]:
                                if content_type not in ["image", "photo"]:
                                    special_content.append(content_type)
                    
                    if special_content:
                        print(f"- Special content identified: {', '.join(set(special_content))}")
                    
                    print("\nYou can now ask questions about the content.")
                else:
                    print("No content could be processed from the provided input.")
            else:
                # This is a question to answer
                if not self.items:
                    print("No content has been processed yet. Please provide a file, URL, or directory to analyze first.")
                else:
                    print("Generating response...")
                    response = self.query_content(user_input)
                    print("\nResponse:")
                    print(response)


class ContentIdentifier:
    """Identifies content types in images"""
    
    def __init__(self, model: AIModel):
        self.model = model
    
    def identify_image_content(self, image: Image.Image, encoded_image: str) -> List[str]:
        """Identify the type of content in an image"""
        return self.model.identify_image_content(image, encoded_image)


def main():
    parser = argparse.ArgumentParser(description='Multimodal AI Agent for Content Understanding')
    parser.add_argument('--input', type=str, help='Path, URL, or directory to analyze')
    parser.add_argument('--model', type=str, default='gemini', choices=['gemini', 'anthropic', 'openai'], 
                        help='AI model to use (default: gemini)')
    parser.add_argument('--api-key', type=str, help='API key for the selected model')
    parser.add_argument('--model-name', type=str, help='Specific model name (optional)')
    parser.add_argument('--verification-model', type=str, choices=['gemini', 'anthropic', 'openai'], 
                        help='Model to use for verification (default: same as primary model)')
    parser.add_argument('--verification-api-key', type=str, help='API key for the verification model')
    parser.add_argument('--verification-model-name', type=str, help='Specific verification model name (optional)')
    
    args = parser.parse_args()
    
    # Configure the primary model
    model_config = {
        "type": args.model,
        "api_key": args.api_key or os.environ.get(f"{args.model.upper()}_API_KEY", "GEMINI_API_KEY"),
        "model_name": args.model_name
    }
    
    # Configure the verification model if specified
    verification_model_config = None
    if args.verification_model:
        verification_model_config = {
            "type": args.verification_model,
            "api_key": args.verification_api_key or os.environ.get(f"{args.verification_model.upper()}_API_KEY"),
            "model_name": args.verification_model_name
        }
    
    # Create agent with specified models
    agent = AIAgent(model_config, verification_model_config)
    
    if args.input:
        agent.process_input(args.input)
    
    agent.run()

if __name__ == "__main__":
    main()
