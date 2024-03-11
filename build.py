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

from embeddings import load_embedding
from dotenv import load_dotenv

import os

load_dotenv(verbose=True)

def create_vector():
    loader = GitLoader(
        clone_url = os.getenv('GIT_CLONE_URL'),
        branch = os.getenv('GIT_BRANCH'),
        repo_path = "data/docs",
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language = Language.PHP,
        chunk_size = 5000,
        chunk_overlap = 3000
    )

    texts = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents = texts,
        persist_directory = "data/vectors",
        embedding = load_embedding()
    )

    vectorstore.persist()

if __name__=="__main__":
    create_vector()