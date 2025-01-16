
from pinecone import Pinecone
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from settings import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "text-embedding-3-small"

#

def get_index(index_name):
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(index_name)
    except Exception as e:
        index = None
    return index

def process_pdf(file_path):
    # create a loader
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [doc.page_content.replace("\n", " ") for doc in documents]
    return texts

# Define a function to create embeddings
def create_embeddings(texts):
    embeddings_list = []
    # itrate in batch of 32 texts
    for i in range(0, len(texts), 32):
        text = texts[i:i+32]
        res = client.embeddings.create(
            input=text,
            model=MODEL
        )

        embeddings_list.extend([embed.embedding for embed in res.data])
    return embeddings_list

# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids, texts, document_id):
    # process embeddings in batch of 1000
    for i in range(0, len(embeddings), 1000):
        vectors = []
        for text, embedding, id in zip(texts[i:i+1000], embeddings[i:i+1000], ids[i:i+1000]):
            metadata = {
                "document_id": document_id,
                "text": text
            }
            vectors.append((id, embedding, metadata))
        response = index.upsert(vectors=vectors, namespace="pdf_books")
        print(response)


