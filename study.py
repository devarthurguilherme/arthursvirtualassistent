import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
import tempfile
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

GROC_API_KEY = os.getenv("GROC_API_KEY")

# Streamlit Configure here
st.set_page_config(page_title="Arthur's Assistant 📚",
                   page_icon="📚", layout='wide')
st.title("Arthur's Assistant 📚")

modelClass = "groc"

# Dicionário com os arquivos PDF
pdf_files = {
    "Power BI Completo": r"C:\Users\arthu\Desktop\ArthurAssistent\data\Power BI Completo - Do Básico ao Avançado.pdf",
    "Manipulação e Análise de Dados com Pandas": r"C:\Users\arthu\Desktop\ArthurAssistent\data\Manipulação e Análise de Dados com Pandas.pdf",
    "SQL": r"C:\Users\arthu\Desktop\ArthurAssistent\data\SQL e MySQL de forma prática e objetiva, e ainda crie projetos com PHP e MySQL.pdf"
}

# Sidebar para seleção de PDFs
selected_pdf = st.sidebar.selectbox(
    "Escolha um PDF para carregar:", list(pdf_files.keys()))

# Obtém o caminho do PDF selecionado
uploads = pdf_files[selected_pdf]


def modelGroc(model="llama3-70b-8192", temperature=0.2):
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=GROC_API_KEY  # Use .env for API key
    )
    return llm


def modelHfHub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
    llm = HuggingFaceHub(
        repo_id=model,
        model_kwargs={
            "temperature": temperature,
            "return_full_text": False,
            "max_new_tokens": 512,
        }
    )
    return llm


def configRetriever(upload):
    docs = []
    loader = PyPDFLoader(upload)
    docs.extend(loader.load())

    # Split Text
    textSplitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    splits = textSplitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Store Documents
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local('vectorstore/db_faiss')

    # Retriver Configure
    retriever = vectorstore.as_retriever(
        search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 4})
    return retriever


def configRagChain(modelClass, retriever):
    if modelClass == "hf_hub":
        llm = modelHfHub()
    elif modelClass == "groc":
        llm = modelGroc()

    # Define Prompt
    token_s, token_e = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>",
                        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>") if modelClass.startswith("hf") else ("", "")

    contextSystemPrompt = token_s + "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Don't answer the question, just reformulate it if needed and otherwise return it as is." + token_e
    contextUserPrompt = "Question: {input}" + token_e
    contextPrompt = ChatPromptTemplate.from_messages([(
        "system", contextSystemPrompt),
        MessagesPlaceholder("chat_history"),
        ("human", contextUserPrompt),
    ])

    # Chain to context
    historyAwareRetriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextPrompt)

    # Prompt Template to Questions and Answers
    questionsAnswersPromptTemplate = """You are a Virtual Assistant very useful and answer general questions.
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


if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm your Virtual Assistant! How can I help you?")]

if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

start = time.time()
userQuery = st.chat_input("Digite sua mensagem aqui...")

if userQuery and uploads:
    st.session_state.chat_history.append(HumanMessage(content=userQuery))

    with st.chat_message("Human"):
        st.markdown(userQuery)

    with st.chat_message("AI"):
        with st.spinner("Loading..."):
            progress = st.progress(0)  # Progress bar

            for i in range(100):
                time.sleep(0.02)  # Simulate processing time
                progress.progress(i + 1)  # Update progress bar

            if st.session_state.docs_list != uploads:
                st.session_state.docs_list = uploads
                st.session_state.retriever = configRetriever(uploads)

            ragChain = configRagChain(modelClass, st.session_state.retriever)
            result = ragChain.invoke(
                {"input": userQuery, "chat_history": st.session_state.chat_history})

            resp = result['answer']
            st.write(resp)

            # Show Information Source
            st.write("### Fontes das Informações:")
            sources = result['context']
            for idx, doc in enumerate(sources):
                source = doc.metadata['source']
                file = os.path.basename(source)
                page = doc.metadata.get('page', 'Página não encontrada')

                # Font 1: doc.pdf - p. 2
                ref = f": link: Fonte {idx}: *{file} - p. {page}"
                with st.popover(ref):
                    st.caption(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=resp))

end = time.time()
print("Tempo: ", end - start)
