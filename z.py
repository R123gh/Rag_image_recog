import streamlit as st 
import chromadb 
from sentence_transformers import SentenceTransformer 
import os 
from dotenv import load_dotenv
from groq import Groq 
import speech_recognition as sr 
from gtts import gTTS 
import tempfile 
import re  
from PIL import Image 
import io 
import ocrspace 
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv() 

st.set_page_config(
    page_title="Rag Based System",
    layout='wide'
) 

if 'text_anser' not in st.session_state:
    st.session_state.text_answer = None
if 'voice_query' not in st.session_state:
    st.session_state.voice_query = None 


def validate_environment():
    required_keys = ['GROQ_API_KEY' , 'OCR_SPACE_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        st.error(f"Missing environment variable : {', '.join(missing_keys)}")
        st.stop() 
validate_environment()
@st.cache_resource
def laod_system():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path="./chroma_db") 
        
        try:
            collection = client.get_collection(name="video_chunks")
        except Exception as e:
            st.error("ChromaDB collection 'video_chunks' not found. please run vector_db.py first." )
            st.stop()
            
            groq_client = Groq(api_key = os.getenv("GROQ_API_KEY"))
            ocr_api = ocrspace.API(api_key=os.getenv("OCR_SPACE_API_KEY"))
            
            logger.info("All system loaded successfully")
            return model , collection , groq_client , ocr_api 
        except Exception as e:
            logger.error(f"Failed to load system : {str(e)}")
            st.error(f"System initialization failed : {str(e)}")
            st.stop()
            
    model , collection , groq_client , ocr_api  = load_dotenv()
    
def search_chunks(query , top_k = 5):
    try:
        query_embedding = model.encode([query])[0].tolist() 
        return collection.query(query_embedding = [query_embedding] , n_results = top_k)
    except Exception as e :
        logger.error(f"Serach")