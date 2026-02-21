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


load_dotenv()

st.set_page_config(page_title="RAG Based System and Voice Assistant", layout="centered")

@st.cache_resource
def load_system():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="video_chunks")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return model, collection, groq_client

model, collection, groq_client = load_system()

if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'stop_recording' not in st.session_state:
    st.session_state.stop_recording = False

def search_chunks(query, top_k=5):
    query_embedding = model.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k 
    )
    return results 

def get_answer(query, top_k=5):
    results = search_chunks(query, top_k)
    
    if not results['documents'][0]:
        return None
    
    context = "\n\n".join(results['documents'][0])
        
    prompt = f"""You are a smart voice assistant helping users understand content from video tutorials.

Video Context:
{context}

User Question: {query}

Instructions:
- Give a clear, natural, conversational answer like a helpful teacher
- Use the video content as your primary source of information
- Explain concepts in simple, easy-to-understand language
- Keep your answer focused and concise (3-5 sentences)
- If using bullet points, make each point complete and meaningful
- Avoid repetitive or fragmented sentences
- Don't mention "the video says" or "according to the video" - just explain naturally
- If the video covers the topic, explain it clearly; if not, give a brief general explanation
- Sound natural and helpful, like you're talking to someone face-to-face
- Give me the result in bullets points and do not add extra information 

Answer:"""
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=500
    )
    
    return response.choices[0].message.content

def record_voice():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
        
        text = r.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except Exception as e:
        return None

def clean_text_for_speech(text):
    cleaned = re.sub(r'[•\-\*]\s*', '', text)
    cleaned = re.sub(r'\*\*(.+?)\*\*', r'\1', cleaned)
    cleaned = re.sub(r'\*(.+?)\*', r'\1', cleaned)
    cleaned = re.sub(r'#{1,6}\s*', '', cleaned)
    cleaned = re.sub(r'\n+', '. ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = re.sub(r'\.\.+', '.', cleaned)
    return cleaned.strip()

def speak_text(text):
    clean_text = clean_text_for_speech(text)
    tts = gTTS(text=clean_text, lang='en', slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton > button {
        height: 50px;
        font-size: 18px;
        border-radius: 8px;
    }
    .listening-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .transcribed-text {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title(" RAG Voice Assistant")
st.markdown("Ask questions using voice or text!")

col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input("", placeholder="Type or use voice...", label_visibility="collapsed", key="text_input")

with col2:
    if not st.session_state.recording:
        voice_button = st.button("🎤", help="Voice Input", use_container_width=True, key="start_voice")
        if voice_button:
            st.session_state.recording = True
            st.rerun()

if st.session_state.recording:
    st.markdown('<div class="listening-container">🎤 Listening... Speak now!</div>', unsafe_allow_html=True)
    
    col_stop1, col_stop2, col_stop3 = st.columns([1, 2, 1])
    with col_stop2:
        if st.button("⏹️ Stop Recording", use_container_width=True, type="primary"):
            st.session_state.recording = False
            st.session_state.stop_recording = True
            st.rerun()
    
    if not st.session_state.stop_recording:
        voice_text = record_voice()
        st.session_state.recording = False
        
        if voice_text:
            st.session_state.voice_query = voice_text
            st.session_state.use_voice = True
            st.markdown(f'<div class="transcribed-text">✓ <strong>You said:</strong> {voice_text}</div>', unsafe_allow_html=True)
            query = voice_text
        else:
            st.error(" Could not understand. Please try again.")
            st.session_state.recording = False
    else:
        st.session_state.stop_recording = False
        st.warning("⏹️ Recording stopped")

if 'voice_query' in st.session_state and not query:
    query = st.session_state.voice_query
    st.markdown(f'<div class="transcribed-text">📝 <strong>Your question:</strong> {query}</div>', unsafe_allow_html=True)

if query:
    with st.spinner(" Thinking..."):
        answer = get_answer(query, top_k=5)
        
        if answer:
            st.markdown("---")
            st.markdown("**💡 Answer:**")
            st.markdown(answer)
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            
            with col_btn1:
                listen_btn = st.button("🔊 Listen", use_container_width=True)
            
            with col_btn2:
                new_btn = st.button("🔄 New", use_container_width=True)
            
            if st.session_state.get('use_voice', False):
                with st.spinner("🔊 Speaking..."):
                    audio_file = speak_text(answer)
                    with open(audio_file, 'rb') as f:
                        st.audio(f.read(), format='audio/mp3', autoplay=True)
                    os.remove(audio_file)
                st.session_state.use_voice = False
            
            if listen_btn:
                with st.spinner("🔊 Generating audio..."):
                    audio_file = speak_text(answer)
                    with open(audio_file, 'rb') as f:
                        st.audio(f.read(), format='audio/mp3', autoplay=True)
                    os.remove(audio_file)
            
            if new_btn:
                for key in ['voice_query', 'use_voice', 'recording', 'stop_recording']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        else:
            st.warning(" No relevant answer found. Try rephrasing.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: #888; font-size: 12px;'>📊 {collection.count()} chunks available</p>", unsafe_allow_html=True)