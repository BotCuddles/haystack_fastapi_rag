# app/pipelines.py

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder, ChatPromptBuilder
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
from haystack.components.joiners import BranchJoiner, DocumentJoiner
from haystack.components.converters import OutputAdapter, TextFileToDocument, MarkdownToDocument, PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.dataclasses import ChatMessage
from app.config import settings
from pathlib import Path
import gdown

# Initialize document store and embedding model
document_store = InMemoryDocumentStore()
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
retriever = InMemoryBM25Retriever(document_store=document_store, top_k=3)

# Document downloading and processing pipeline
def download_and_process_documents():
    url = "https://drive.google.com/drive/folders/1zLGOQEPNUlTKlsJdKk7E4x21WUJI7pfV?usp=sharing"
    output_dir = settings.DOCUMENT_STORE_DIR
    gdown.download_folder(url, quiet=True, output=output_dir)

    # Define converters, preprocessors, and joiner
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
    text_file_converter = TextFileToDocument()
    markdown_converter = MarkdownToDocument()
    pdf_converter = PyPDFToDocument()
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=250, split_overlap=50)
    document_joiner = DocumentJoiner()
    document_writer = DocumentWriter(document_store)

    # Set up preprocessing pipeline
    preprocessing_pipeline = Pipeline()
    preprocessing_pipeline.add_component(instance=file_type_router, name="file_type_router")
    preprocessing_pipeline.add_component(instance=text_file_converter, name="text_file_converter")
    preprocessing_pipeline.add_component(instance=markdown_converter, name="markdown_converter")
    preprocessing_pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
    preprocessing_pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    preprocessing_pipeline.add_component(instance=document_splitter, name="document_splitter")
    preprocessing_pipeline.add_component(instance=document_embedder, name="document_embedder")
    preprocessing_pipeline.add_component(instance=document_writer, name="document_writer")
    preprocessing_pipeline.add_component(instance=document_joiner, name="document_joiner")

    # Connect components in the preprocessing pipeline
    preprocessing_pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
    preprocessing_pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
    preprocessing_pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
    preprocessing_pipeline.connect("text_file_converter", "document_joiner")
    preprocessing_pipeline.connect("pypdf_converter", "document_joiner")
    preprocessing_pipeline.connect("markdown_converter", "document_joiner")
    preprocessing_pipeline.connect("document_joiner", "document_cleaner")
    preprocessing_pipeline.connect("document_cleaner", "document_splitter")
    preprocessing_pipeline.connect("document_splitter", "document_embedder")
    preprocessing_pipeline.connect("document_embedder", "document_writer")

    # Run preprocessing
    preprocessing_pipeline.run({"file_type_router": {"sources": list(Path(output_dir).glob("**/*"))}})

# Initialize RAG components
query_rephrase_llm = HuggingFaceAPIGenerator(api_type="serverless_inference_api", api_params={"model": "mistralai/Mistral-7B-Instruct-v0.3", "max_new_tokens": 350})
llm = HuggingFaceAPIChatGenerator(api_type="serverless_inference_api", api_params={"model": "mistralai/Mistral-7B-Instruct-v0.3", "max_new_tokens": 350})

# Create conversational RAG pipeline
def create_conversational_rag_pipeline(memory_store):
    conversational_rag = Pipeline()
    
    conversational_rag.add_component("query_rephrase_prompt_builder", PromptBuilder(query_rephrase_template))
    conversational_rag.add_component("query_rephrase_llm", query_rephrase_llm)
    conversational_rag.add_component("list_to_str_adapter", OutputAdapter(template="{{ replies[0] }}", output_type=str))
    
    conversational_rag.add_component("retriever", retriever)
    conversational_rag.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents", "memories"], required_variables=["query", "documents", "memories"]))
    conversational_rag.add_component("llm", llm)
    
    conversational_rag.add_component("memory_retriever", memory_store)
    conversational_rag.add_component("memory_writer", memory_store)
    conversational_rag.add_component("memory_joiner", BranchJoiner(list[ChatMessage]))

    conversational_rag.connect("memory_retriever", "query_rephrase_prompt_builder.memories")
    conversational_rag.connect("query_rephrase_prompt_builder.prompt", "query_rephrase_llm")
    conversational_rag.connect("query_rephrase_llm.replies", "list_to_str_adapter")
    conversational_rag.connect("list_to_str_adapter", "retriever.query")
    conversational_rag.connect("retriever.documents", "prompt_builder.documents")
    conversational_rag.connect("prompt_builder.prompt", "llm.messages")
    conversational_rag.connect("llm.replies", "memory_joiner")
    conversational_rag.connect("memory_joiner", "memory_writer")
    conversational_rag.connect("memory_retriever", "prompt_builder.memories")
    
    return conversational_rag
