from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, GitLoader
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings import OllamaEmbeddings  

import os

def create_vector_db():
    loader = GitLoader(
        clone_url = "https://github.com/example/example",
        branch = "development",
        repo_path = "repo",
    )

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PHP,
        chunk_size=5000,
        chunk_overlap=3000
    )
    
    texts = text_splitter.split_documents(documents)
    
    vectorstore = Chroma.from_documents(
        documents = texts,
        embedding = GPT4AllEmbeddings(),
        persist_directory = "vectorstores/db/"
    )

    vectorstore.persist()

if __name__=="__main__":
    create_vector_db()