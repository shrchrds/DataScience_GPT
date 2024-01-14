from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

# initialize Pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")


extracted_Data = load_pdf("data/")
text_chunks = text_split(extracted_Data)
embeddings = download_hugging_face_embeddings()

# initialize Pinecone

pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV
)
# give your own pine index name
index_name = "datascience-gpt"

# createing embeddings for each of the text chunks & store
docsearch = Pinecone.from_texts(
    [t.page_content for t in text_chunks], embeddings, index_name=index_name
)