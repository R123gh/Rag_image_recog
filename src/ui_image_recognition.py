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
    page_title="RAG Based System",
    layout="wide"
)

if 'text_answer' not in st.session_state:
    st.session_state.text_answer = None
if 'voice_query' not in st.session_state:
    st.session_state.voice_query = None

def validate_environment():
    required_keys = ['GROQ_API_KEY', 'OCR_SPACE_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        st.error(f"Missing environment variables: {', '.join(missing_keys)}")
        st.stop()

validate_environment()

@st.cache_resource
def load_system():
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            collection = client.get_collection(name="video_chunks")
        except Exception as e:
            st.error(" ChromaDB collection 'video_chunks' not found. Please run vector_db.py first.")
            st.stop()
        
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        ocr_api = ocrspace.API(api_key=os.getenv("OCR_SPACE_API_KEY"))
        
        logger.info("✓ All systems loaded successfully")
        return model, collection, groq_client, ocr_api
    
    except Exception as e:
        logger.error(f"Failed to load system: {str(e)}")
        st.error(f" System initialization failed: {str(e)}")
        st.stop()

model, collection, groq_client, ocr_api = load_system()

def search_chunks(query, top_k=5):
    try:
        query_embedding = model.encode([query])[0].tolist()
        return collection.query(query_embeddings=[query_embedding], n_results=top_k)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {'documents': [[]], 'metadatas': [[]]}

def get_answer_text_only(query, top_k=5):
    if not query or len(query.strip()) == 0:
        return "Please provide a valid question."
    
    try:
        results = search_chunks(query, top_k)
        
        if not results['documents'][0]:
            return "No relevant information found in the knowledge base."
        
        context = "\n\n".join(results['documents'][0])
        
        prompt = f"""You are a helpful AI assistant answering questions about video content.

Video Content:
{context}

Question: {query}

Instructions:
- Use information from the video content when available
- If the video mentions related concepts, use them to explain the topic
- Combine the video content with your general knowledge to provide precise answers
- Do not add the extra information only dependent on the text 
- Be positive and constructive - focus on what you CAN explain
- If the exact term isn't defined in the video but related concepts are mentioned, explain the concept using both the video context and general knowledge in less manner
- Never say "the video doesn't cover this" - instead, provide relevant information from the video
- Explain in topic five to six lines and use bullet points if needed to make it easier to read 
- For all the questions, try to use text from the videos as much as possible 
- Give me answer only one time and do not repeat the answer in different ways 
- It should handle all type of questions and give me similar answers for all asked answer
- If the video text is not present so explain the concept in precise 
-Always provide the informative results 

Provide a clear and helpful answer:"""
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Answer generation error: {str(e)}")
        return f"Error generating answer. Please try again."

def process_image(uploaded_file):
    try:
        uploaded_file.seek(0)
        original_bytes = uploaded_file.read()
        
        if len(original_bytes) == 0:
            raise ValueError("Empty image file")
        
        image = Image.open(io.BytesIO(original_bytes))
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=85)
        img_bytes = img_byte_arr.getvalue()
        
        return image, img_bytes, original_bytes
    
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

def get_combined_answer(image_text, query, top_k=5):
    try:
        search_query = f"{image_text} {query}" if query else image_text
        results = search_chunks(search_query, top_k)
        video_context = "\n\n".join(results['documents'][0]) if results['documents'][0] else ""
        
        prompt = f"""You are a helpful AI assistant that combines information from multiple sources.

TEXT EXTRACTED FROM IMAGE (via OCR):
{image_text}

RELATED CONTENT FROM VIDEO DATABASE:
{video_context if video_context else "No related video content found."}

USER QUESTION: {query if query else "Explain what's in the image and provide relevant information."}

INSTRUCTIONS:
- Analyze the text extracted from the image
- Use the related video content to provide additional context
- Answer the user's question in a precise manner
- Be clear, detailed, and helpful
- If the image contains code, explain it step by step
- If there's related information in the video database, mention it
- Combine all sources naturally in your response
- Explain in 3-4 lines and use bullet points if needed
- For all questions, try to use text from the videos as much as possible
- Give answer only one time and do not repeat in different ways
- If video text is not present, explain the concept precisely

Provide a comprehensive answer:"""
        
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
            timeout=30
        )
        
        return response.choices[0].message.content, video_context, results
    
    except Exception as e:
        logger.error(f"Combined answer error: {str(e)}")
        return f"Error generating answer: {str(e)}", "", {'documents': [[]], 'metadatas': [[]]}

def record_voice():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
        return r.recognize_google(audio)
    except sr.WaitTimeoutError:
        return None
    except sr.UnknownValueError:
        return None
    except Exception as e:
        logger.error(f"Voice recognition error: {str(e)}")
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
    try:
        if not text or len(text.strip()) == 0:
            raise ValueError("No text to convert to speech")
        
        clean_text = clean_text_for_speech(text)
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    
    except Exception as e:
        logger.error(f"Text-to-speech error: {str(e)}")
        raise

st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton > button {
        border-radius: 8px;
        padding: 10px 20px;
    }
    .ocr-result {
        background: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 15px 0;
    }
    .video-context {
        background: #d1ecf1;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 15px 0;
    }
    .llm-response {
        background: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 15px 0;
    }
    .result-section {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        margin: 15px 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

st.title(" RAG Based system ")

tab1, tab2 = st.tabs(["💬 Text/Voice Chat", "📷 Image Analysis"])

with tab1:
    col1, col2 = st.columns([5, 1])
    
    with col1:
        query = st.text_input(
            "",
            placeholder="Type your question here...",
            label_visibility="collapsed",
            key="text_input",
            max_chars=500
        )
    
    with col2:
        voice_btn = st.button("🎤", help="Voice Input", use_container_width=True, key="voice_btn")
    
    if voice_btn:
        with st.spinner("🎤 Listening..."):
            voice_text = record_voice()
            if voice_text:
                st.session_state.voice_query = voice_text
                st.success(f"✓ {voice_text}")
                query = voice_text
            else:
                st.error("Could not understand.")
    
    if st.session_state.voice_query and not query:
        query = st.session_state.voice_query
    
    if query:
        with st.spinner(" Thinking..."):
            answer = get_answer_text_only(query, top_k=5)
            st.session_state.text_answer = answer
        
        if answer and not answer.startswith("Error"):
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)
            
            col_btn1, col_btn2 = st.columns([1, 1])
            
            with col_btn1:
                listen_btn = st.button("🔊 Listen", key="listen_text_btn", use_container_width=True)
            
            with col_btn2:
                if st.button("🔄 Clear", key="clear_text_btn", use_container_width=True):
                    st.session_state.voice_query = None
                    st.session_state.text_answer = None
                    st.rerun()
            
            if listen_btn and st.session_state.text_answer:
                try:
                    with st.spinner("🔊 Generating audio..."):
                        audio_file = speak_text(st.session_state.text_answer)
                        with open(audio_file, 'rb') as f:
                            st.audio(f.read(), format='audio/mp3', autoplay=False)
                        os.remove(audio_file)
                except Exception as e:
                    st.error(f" Audio generation failed: {str(e)}")
        else:
            st.error(answer)

with tab2:
    uploaded_file = st.file_uploader(
        "Upload image",
        type=['png', 'jpg', 'jpeg', 'webp', 'gif'],
        key="upload_ocr"
    )
    
    if uploaded_file:
        try:
            col_img, col_controls = st.columns([2, 1])
            
            with col_img:
                with st.spinner("📷 Processing..."):
                    display_image, img_bytes, original_bytes = process_image(uploaded_file)
                st.image(display_image, caption="Uploaded Image", width=250)
            
            with col_controls:
                optional_query = st.text_area(
                    "Your Question (Optional):",
                    placeholder="Ask specific question...",
                    height=120,
                    key="optional_query",
                    max_chars=300
                )
                
                ocr_engine = st.selectbox(
                    "OCR Engine:",
                    options=[1, 2],
                    index=1
                )
                
                analyze_button = st.button(
                    "🚀 Analyze",
                    type="primary",
                    use_container_width=True
                )
            
            if analyze_button:
                st.markdown("---")
                
                with st.spinner("📝 Extracting text..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            tmp.write(original_bytes)
                            tmp_path = tmp.name
                        
                        extracted_text = ocr_api.ocr_file(tmp_path)
                        os.remove(tmp_path)
                        
                        if not extracted_text or len(extracted_text.strip()) == 0:
                            extracted_text = "No text found in image"
                    
                    except Exception as e:
                        logger.error(f"OCR error: {str(e)}")
                        extracted_text = f"Error: {str(e)}"
                
                st.markdown('<div class="ocr-result">', unsafe_allow_html=True)
                st.markdown("**Extracted Text:**")
                
                if extracted_text.startswith("Error"):
                    st.error(extracted_text)
                else:
                    st.text_area("", extracted_text, height=150, key="ocr_output", label_visibility="collapsed")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                if not extracted_text.startswith("Error") and extracted_text != "No text found in image":
                    with st.spinner("🎥 Searching database..."):
                        llm_answer, video_context, search_results = get_combined_answer(
                            extracted_text, 
                            optional_query, 
                            top_k=5
                        )
                    
                    st.markdown('<div class="video-context">', unsafe_allow_html=True)
                    st.markdown("**Related Content:**")
                    
                    if video_context:
                        with st.expander("View Related Chunks", expanded=False):
                            for i, (doc, meta) in enumerate(zip(
                                search_results['documents'][0], 
                                search_results['metadatas'][0]
                            )):
                                st.markdown(f"**Chunk {i+1}**")
                                st.text(doc[:200] + "..." if len(doc) > 200 else doc)
                                st.markdown("---")
                    else:
                        st.info("No related content found")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<div class="llm-response">', unsafe_allow_html=True)
                    st.markdown("**Answer:**")
                    st.markdown(llm_answer)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    full_result = f"""=== IMAGE OCR TEXT ===
{extracted_text}

=== RELATED VIDEO CONTENT ===
{video_context if video_context else "No related content found"}

=== AI ANSWER ===
{llm_answer}
"""
                    st.download_button(
                        "📥 Download",
                        full_result,
                        file_name="result.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
        
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            st.error(f" {str(e)}")