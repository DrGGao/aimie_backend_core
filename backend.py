import os
import fitz  # For PDF parsing
import nltk
import docx2txt
from typing import Optional, List, Dict, Any
from nltk.tokenize import sent_tokenize
import logging
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Form, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],                  
    allow_headers=["*"],                  
)

import uvicorn
from openai import embeddings

# ----------------- Logging Configuration -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- AstraDB Configuration -----------------
from astrapy import DataAPIClient
from astrapy.info import CollectionVectorServiceOptions

# Define Keyspace and Collection names
KEYSPACE = "testv4"
# current_date = datetime.now().strftime("%Y%m%d")
# COLLECTION_NAME = "dstc" + current_date
COLLECTION_NAME = "dstc20240306"

logger.info("Query Result:")
logger.info("COLLECTION_NAME: " + COLLECTION_NAME)

# AstraDB constant configuration
ASTRA_DB_API_ENDPOINT = "https://***********.apps.astra.datastax.com"
ASTRA_DB_APPLICATION_TOKEN = "***********"

# ----------------- Download nltk Data -----------------
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ----------------- Global Parameters -----------------
split_size = 15   # Number of sentences per chunk
overlap = 1       # Number of overlapping sentences
top_k_docs = 10   # Number of documents to return during retrieval

# ----------------- Environment Variables Configuration -----------------
os.environ["OPENAI_API_KEY"] = "sk-proj-**********************"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-***********-***********"
os.environ['AWS_ACCESS_KEY_ID'] = "***********"
os.environ['AWS_SECRET_ACCESS_KEY'] = "***********+"
os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
os.environ["GOOGLE_API_KEY"] = "***********"
os.environ["XAI_API_KEY"] = "xai-***********"
os.environ["DEEPSEEK_API_KEY"] = "sk-***********"

# ----------------- Model Configuration -----------------
CONFIG = {
    "provider": "openai",
    "models": {
        "openai": {
            "gpt-4": {"temperature": 0.7, "max_retries": 10},
            "gpt-3.5": {"temperature": 0.7, "max_retries": 10},
            "gpt-3.5-turbo": {"temperature": 0.7, "max_retries": 10},
            "gpt-4o-mini": {"temperature": 0.7, "max_retries": 10},
        },
        "anthropic": {
            "claude-3-7-sonnet-20250219": {"temperature": 0.7},
            "claude-3-5-sonnet-20241022": {"temperature": 0.7},
            "claude-3-5-haiku-20241022": {"temperature": 0.7},
        },
        # "awsbedrock": {
        #     "meta.llama3-3-70b-instruct-v1:0": {"temperature": 0.7},
        #     "meta.llama3-2-90b-instruct-v1:0": {"temperature": 0.7},
        #     "meta.llama3-2-1b-instruct-v1:0": {"temperature": 0.7},
        # },
        "google": {
            "gemini-1.5-pro": {"temperature": 0, "max_retries": 10},
            "gemini-1.5-flash": {"temperature": 0, "max_retries": 10},
            "gemini-2.0-flash-exp": {"temperature": 0, "max_retries": 10},
        },
        # "xai": {
        #     "grok-beta": {"temperature": 0, "max_retries": 10},
        #     "grok-2-latest": {"temperature": 0, "max_retries": 10},
        # },
        "deepseek":{
            "deepseek-reasoner":{"max_retries":20},
            "deepseek-chat":{"max_retries":20}
        }
    }
}

# ----------------- Utility Functions -----------------

def get_llm(provider: Optional[str] = None, model_name: str = None):
    """
    Generate the corresponding LLM instance based on the provider and model name.
    """
    if provider is None:
        provider = CONFIG["provider"]
    if provider not in CONFIG['models']:
        raise ValueError(f"Unsupported provider: {provider}")
    if model_name is None:
        model_name = list(CONFIG['models'][provider].keys())[0]
    if model_name not in CONFIG['models'][provider]:
        raise ValueError(f"Unsupported model: {model_name} for provider: {provider}")
    params = CONFIG["models"][provider][model_name]

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, **params)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, **params)
    elif provider == "awsbedrock" or provider == "AWS" or provider == "aws":
        from langchain_aws import ChatBedrock
        return ChatBedrock(model_id=model_name, model_kwargs=params, region="us-east-1")
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, **params)
    elif provider == "xai":
        from langchain_xai import ChatXAI
        return ChatXAI(model=model_name, **params)
    elif provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=model_name, **params)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def chunk_by_bytes(sentences: List[str], max_bytes: int = 7000) -> List[str]:
    """
    Split the text by bytes, ensuring each chunk does not exceed max_bytes.
    """
    chunks = []
    current_chunk = []
    current_size = 0

    for sentence in sentences:
        sentence_bytes = len(sentence.encode('utf-8'))
        if sentence_bytes > max_bytes:
            # If a single sentence itself is larger than max_bytes,
            # we force it as a separate chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            chunks.append(sentence)
            continue
        if current_size + sentence_bytes > max_bytes:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_bytes
        else:
            current_chunk.append(sentence)
            current_size += sentence_bytes

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def load_from_text(filename: str, text: str, split_size: int, overlap: int):
    """
    Split the text into sentences, merge sentences based on split_size and overlap,
    and then further split the merged text by bytes.
    """
    sentences = sent_tokenize(text)
    i = 0
    merged_data = []
    while i < len(sentences):
        chunk_sentences = sentences[i:i + split_size]
        merged_data.append(" ".join(chunk_sentences))
        i = i + (split_size - overlap)

    final_chunks = []
    for segment in merged_data:
        segment_sentences = [s.strip() for s in segment.split('.') if s.strip()]
        segment_sentences = [s + '.' for s in segment_sentences]
        chunked_by_bytes_list = chunk_by_bytes(segment_sentences, max_bytes=7000)
        for c in chunked_by_bytes_list:
            final_chunks.append(f"{filename} {c}")
    return final_chunks

def extract_text_from_pdf(file_path: str):
    """
    Extract text from a PDF file.
    """
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(file_path: str):
    """
    Extract text from a DOCX file.
    """
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path: str):
    """
    Extract text from a TXT file.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def load_and_chunk_docs(paths: List[str], split_size: int, overlap: int):
    """
    Extract and split text from each file, returning all split text data.
    """
    chunked_datas = []
    for p in paths:
        lower_p = p.lower()
        if lower_p.endswith(".pdf"):
            text = extract_text_from_pdf(p)
        elif lower_p.endswith(".docx"):
            text = extract_text_from_docx(p)
        elif lower_p.endswith(".txt"):
            text = extract_text_from_txt(p)
        else:
            logger.warning(f"Unsupported file type: {p}, skipping.")
            continue
        chunks = load_from_text(os.path.basename(p), text, split_size, overlap)
        chunked_datas.extend(chunks)
    return chunked_datas

def query_system(chain, query, history):
    """
    """
    result = chain.invoke({"input": query})
    history.append(("human", query))
    history.append(("system", result["answer"]))
    return result

def print_query_result(result,strr):
    logger.info(strr)
    logger.info("Query Result:")
    logger.info(f"Input: {result['input']}")
    logger.info(f"Answer: {result['answer']}")

def print_conversation_history(history):
    logger.info("Conversation History:")
    for index, (speaker, text) in enumerate(history):
        logger.info(f"{index + 1}. {speaker.title()}: {text}")


from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore

def build_database_with_files(
        files: List[str],
        keyspace: str,
        collection_name: str,
        embedding_model: str = "text-embedding-3-small"
):
    """

    """
    # ------------------------------------------------------
    # ------------------------------------------------------
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    database = client.get_database(ASTRA_DB_API_ENDPOINT)
    my_db_admin = database.get_database_admin()

    if keyspace not in my_db_admin.list_keyspaces():
        my_db_admin.create_keyspace(keyspace)
        print(f"Created keyspace: {keyspace}")
        print("Current Keyspace list:", my_db_admin.list_keyspaces())
    else:
        print("Keyspace already exists:", keyspace)
        print("Current Keyspace list:", my_db_admin.list_keyspaces())

    db = client.get_database(
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        keyspace=keyspace
    )
    if collection_name not in db.list_collection_names():
        db.create_collection(
            name=collection_name,
            dimension=1536, 
        )
        print(f"Created collection: {collection_name}")
        print("Current Collection list:", db.list_collection_names())
    else:
        print("Collection already exists:", collection_name)
        print("Current Collection list:", db.list_collection_names())

    # ------------------------------------------------------
    # ------------------------------------------------------
    logger.info("Loading and chunking documents...")
    documents = load_and_chunk_docs(files, split_size, overlap)
    if not documents:
        logger.error("No documents found or processed.")
        return {"error": "No documents to process."}

    # ------------------------------------------------------
    # ------------------------------------------------------
    logger.info("Initializing embeddings and vectorstore...")
    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectorstore = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name=collection_name,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=keyspace,
    )

    logger.info("Adding documents to the AstraDB VectorStore...")
    vectorstore.add_texts(documents)
    logger.info("Documents added successfully.")

    return {"message": "Database build finished."}


def get_retrieval_chain(
        keyspace: str,
        collection_name: str,
        provider: str = "openai",
        model_name: str = "gpt-4",
        search_k: int = 10
):
    """

    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = AstraDBVectorStore(
        embedding=embeddings,
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        collection_name=collection_name,
        token=ASTRA_DB_APPLICATION_TOKEN,
        namespace=keyspace,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": search_k}
    )

    llm = get_llm(provider, model_name)

    system_prompt = (
        "You are an academic assistant for the AIMAHEAD initiative. Answer questions using the context provided, and if you are unsure, say you do not know. "
        "Never recommend materials or resources outside of AIMAHEAD."
        "The document is related to the AIMAHEAD course, each course have several modules, and each module have several lessons. "
        "You are a specialized academic assistant capable of discerning the user’s intent and providing relevant answers "
        "related to the AIMAHEAD initiative. When a user asks a question, determine which category it falls into: "
        "1) Course Recommendations, 2) Program Information, 3) Research Query, or 4) Other. "
        "If the user’s request does not match any category, treat it as 'Other.' "
        "When responding, use the context provided. "
        "• If the user’s query is about courses, list over 2 recommended courses from basic to advanced, with brief explanations. And tell user the course list is from basic to advanced.  "
        "• If the query is about program information, such as seminars, provide the latest known details. "
        "• If the query is about research, especially regarding responsible AI or health equity, summarize current findings "
        "and references where possible. "
        "• For all other requests, answer with the best available information or state if you do not know. "
        "Always respond in English if the user’s question is in English, and in Chinese if the user’s question is in Chinese. "
        "Maintain multi-turn context within the same session, and if uncertain or lacking information, say you do not know."
        "Use no more than five sentences and keep the answer concise. "
        "If requested to provide an unsupported format, please use Markdown format. "
        "if the answer is related to an course, must provide the URL "
        "important!!!provide all related URL!!!"
        "Context: {context}"
    )

    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    return chain

# -------------------------------------------------------------
# -------------------------------------------------------------


session_histories: Dict[str, List] = {}
session_chains: Dict[str, Any] = {} 


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/recommend_config")
def recommend_config():
    """
    """
    recommended_keyspace = KEYSPACE
    recommended_collection = COLLECTION_NAME


    recommended_provider = "anthropic"
    recommended_model = "claude-3-7-sonnet-20250219"

    recommended = {
        "keyspace": recommended_keyspace,
        "collection_name": recommended_collection,
        "provider": recommended_provider,
        "model_name": recommended_model,
    }

    return JSONResponse(content={"status": "ok", "recommend": recommended})

@app.get("/accepted_models")
def accepted_models():
    """
    """
    accepted_models = {}
    for provider, model_dict in CONFIG["models"].items():
        accepted_models[provider] = list(model_dict.keys())
    return JSONResponse(content={"status": "ok", "accepted_models": accepted_models})

@app.post("/build_database")
async def build_database_endpoint(
        request: Request,
        keyspace: str = Form(...),
        collection_name: str = Form(...),
        files: List[UploadFile] = File(...),
        embedding_model: str = "text-embedding-3-small"

):
    """

    """
    saved_paths = []
    for file in files:
        file_location = f"./temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        saved_paths.append(file_location)

    result = build_database_with_files(
        files=saved_paths,
        keyspace=keyspace,
        collection_name=collection_name,
        embedding_model=embedding_model
    )

    return JSONResponse(content={"status": "ok", "detail": result})

@app.post("/query")
async def query_endpoint(
        request: Request,
        query: str = Body(..., embed=True),
        keyspace: str = Body(..., embed=True),
        collection_name: str = Body(..., embed=True),
        provider: str = Body("openai", embed=True),
        model_name: str = Body("gpt-4", embed=True),
        session_id: str = Body("default_session", embed=True),
        top_k: int = Body(10, embed=True)
):
    """

    """
    if session_id not in session_histories:
        session_histories[session_id] = []

    chain_key = f"{keyspace}::{collection_name}::{provider}::{model_name}"
    if chain_key not in session_chains:
        chain = get_retrieval_chain(
            keyspace=keyspace,
            collection_name=collection_name,
            provider=provider,
            model_name=model_name,
            search_k=top_k
        )
        session_chains[chain_key] = chain

    chain = session_chains[chain_key]
    history = session_histories[session_id]

    result = query_system(chain, query, history)
    print_query_result(result,query+" "+keyspace+" "+collection_name+" "+provider+" "+model_name+" "+session_id+" "+str(top_k))

    print_conversation_history(history)
    session_histories[session_id] = history

    return JSONResponse(
        content={
            "query": query,
            "answer": result["answer"],
            "history": history
        }
    )

if __name__ == "__main__":
    client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
    database = client.get_database(ASTRA_DB_API_ENDPOINT)
    my_db_admin = database.get_database_admin()

    if KEYSPACE not in my_db_admin.list_keyspaces():
        my_db_admin.create_keyspace(KEYSPACE)
        print(f"Created keyspace: {KEYSPACE}")
        print("Current Keyspace list:", my_db_admin.list_keyspaces())
    else:
        print("Keyspace already exists:", KEYSPACE)
        print("Current Keyspace list:", my_db_admin.list_keyspaces())

    # Get the database instance for the specified Keyspace
    db = client.get_database(
        api_endpoint=ASTRA_DB_API_ENDPOINT,
        keyspace=KEYSPACE
    )


    # Check if the Collection exists; if not, create it
    if COLLECTION_NAME not in db.list_collection_names():
        db.create_collection(
            name=COLLECTION_NAME,
            dimension=1536,
        )
        print(f"Created collection: {COLLECTION_NAME}")
        print("Current Collection list:", db.list_collection_names())
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")
        print("Current Collection list:", db.list_collection_names())

    # FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)
