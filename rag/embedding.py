from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, GitLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

import os

def load_embedding():
    underlying_embeddings = HuggingFaceEmbeddings(
        model_name = "Salesforce/SFR-Embedding-Mistral"
    )

    store = LocalFileStore("data/cache")
    
    return CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store, 
        namespace=underlying_embeddings.model
    )