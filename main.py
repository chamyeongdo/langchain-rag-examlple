from langchain import hub
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain

import chainlit as cl

import os

from rag import load_embedding
from dotenv import load_dotenv

load_dotenv(verbose=True)

def load_llm():
    llm = Ollama(
        base_url = os.getenv('OLLAMA_BASE_URL'),
        model = os.getenv('OLLAMA_MODEL'),
        verbose = True,
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm


def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever = vectorstore.as_retriever(),
        chain_type_kwargs = {
            "prompt": hub.pull("rlm/rag-prompt-mistral")
        },
        return_source_documents = True,
    )
    return qa_chain

def qa_bot():
    llm = load_llm()
    vectorstore = Chroma(
        persist_directory = "vectorstore",
        embedding_function = load_embedding()
    )
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa

@cl.on_chat_start
async def start():
    chain = qa_bot()
    message = cl.Message(content = "Firing up the research info bot...")
    await message.send()
    message.content = "Hi, welcome to research info bot. What is your query?"
    await message.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    
    callback = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True,
        answer_prefix_tokens = ["FINAL", "ANSWER"]
    )
    callback.answer_reached = True

    response = await chain.acall(
        message.content,
        callbacks = [callback]
    )

    answer = response["result"]
    answer = answer.replace(".",".\n")
    # sources = response["source_documents"]

    # if sources:
    #     answer += f"\nSources: "+str(str(sources))
    # else:
    #     answer += f"\nNo Sources found"

    await cl.Message(content = answer).send()
