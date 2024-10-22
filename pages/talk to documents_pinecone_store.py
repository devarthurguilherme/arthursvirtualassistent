# pip install streamlit
# pip install -U langchain langchain-community
# pip install langchain-huggingface langchain_ollama langchain_openai
# python.exe -m pip install --upgrade pip

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_groq import ChatGroq

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

import torch
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceHub

import io
import tempfile
import os
import time

from langchain_community.vectorstores import Pinecone
import pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyMuPDFLoader

from dotenv import load_dotenv

load_dotenv()
GROC_API_KEY = os.getenv("GROC_API_KEY")

# Streamlit Configure

st.set_page_config(page_title="Talk to Documents 📚",
                   page_icon="📚", layout='wide')

st.title("Talk to Documents 📚")

modelClass = "groc"


def modelGroc(model="llama3-70b-8192", temperature=0.2):
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=GROC_API_KEY
    )
    return llm


def modelHfHub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    # Model Provider
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 512,
            # "stop": ["<|eot_id|>"],
            # other parameters
        }
    )
    return llm


def configRetriever(uploads):
    # Index and Recovery
    docs = []

    for file in uploads:
        # Cria um arquivo temporário para armazenar o PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Lê o conteúdo do arquivo PDF em bytes
            temp_file.write(file.read())
            temp_file.flush()  # Certifica-se de que o conteúdo é gravado no disco
            temp_file_path = temp_file.name  # Guarda o nome do arquivo temporário

        # Agora que o arquivo temporário foi criado e fechado, podemos carregá-lo
        loader = PyMuPDFLoader(temp_file_path)
        docs.extend(loader.load())

        # Remover o arquivo temporário após o uso
        os.remove(temp_file_path)

    # Split Text
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = textSplitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings()

    # Definindo o nome do índice no Pinecone
    index_name = 'outroindex'

    # Criar armazenamento de vetores no Pinecone
    vectorstore = Pinecone.from_documents(
        splits, embeddings, index_name=index_name)

    # Configurando o retriever com base no Pinecone
    retriever = vectorstore.as_retriever(
        search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 4})
    return retriever


def configRagChain(modelClass, retriever):
    # LLM Load
    if modelClass == "hf_hub":
        llm = modelHfHub()
    elif modelClass == "groc":
        llm = modelGroc()

    # Define Prompt
    if modelClass.startswith("hf"):
        token_s, token_e = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>", "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        token_s, token_e = "", ""

    # query -> retriever
    # (query, chatHistoric) -> LLM -> resumedQuery -> retriever
    contextSystemPrompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Don't answer the question, just reformulate it if needed and otherwise return it as is."
    contextSystemPrompt = token_s + contextSystemPrompt
    contextUserPrompt = "Question: {input}" + token_e
    contextPrompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextSystemPrompt),
            MessagesPlaceholder("chat_history"),
            ("human", contextUserPrompt),
        ]
    )

    # Chain to context
    historyAwareRetriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=contextPrompt
    )

    # Prompt Template to Questions and Answers
    questionsAnswersPromptTemplate = """You are a Virtual Assistent very useful and answer general questions.
        Use the following pieces of context to answer the question. 
        If you don't know the answer, just say you don't know. Keep your answer concise..
        Answer in Portuguese. \n\n
        Ask: {input} \n
        Context: {context}"""

    questionsAnswersPrompt = PromptTemplate.from_template(
        token_s + questionsAnswersPromptTemplate + token_e)

    # Configure LLM and Chain for Questions and Answers (Q&A)
    questionAnswerChain = create_stuff_documents_chain(
        llm, questionsAnswersPrompt)
    ragChain = create_retrieval_chain(
        historyAwareRetriever, questionAnswerChain)

    return ragChain


# Sidebar Stremlit
uploads = st.sidebar.file_uploader(
    label="Send File(s)", type=["pdf"],
    accept_multiple_files=True
)
if not uploads:
    st.info("Please, send a file to continue")
    st.stop()

if "llm_chat_history" not in st.session_state:
    st.session_state.llm_chat_history = [
        AIMessage(content="Hello, I'm your Virtual Assistent! How can I help you?"),
    ]


if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

for message in st.session_state.llm_chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

start = time.time()
userQuery = st.chat_input("Digit your message here...")

if userQuery is not None and userQuery != "" and uploads is not None:
    st.session_state.llm_chat_history.append(HumanMessage(content=userQuery))

    with st.chat_message("Human"):
        st.markdown(userQuery)

    with st.chat_message("AI"):
        # Add progress bar
        with st.spinner("Loading..."):
            progress = st.progress(0)  # progress bar

            # Simulate progress while load answer
            for i in range(100):
                time.sleep(0.02)  # Simulate processing time
                progress.progress(i + 1)  # Update progress abr
            if st.session_state.docs_list != uploads:
                st.session_state.docs_list = uploads
                st.session_state.retriever = configRetriever(uploads)

            ragChain = configRagChain(modelClass, st.session_state.retriever)
            result = ragChain.invoke(
                {"input": userQuery, "chat_history": st.session_state.llm_chat_history})

            resp = result['answer']
            st.write(resp)

            # Show Information Source
            sources = result['context']
            for idx, doc in enumerate(sources):
                source = doc.metadata['source']
                file = os.path.basename(source)
                page = doc.metadata.get('page', 'Page not found')

                # Font 1: doc.pdf - p. 2
                ref = f": link: Fonte {idx}: *{file} - p. {page}"
                with st.popover(ref):
                    st.caption(doc.page_content)

    st.session_state.llm_chat_history.append(AIMessage(content=resp))

end = time.time()
print("Tempo: ", end - start)
