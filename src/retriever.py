import chromadb 
from sentence_transformers import SentenceTransformer
import os 
from dotenv import load_dotenv 
from groq import Groq 

load_dotenv()

print("Loading system...")

model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded")

print("Connecting to existing ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

try:
    collection = client.get_collection(name="video_chunks")
    print(f"✓ Connected to existing collection: video_chunks")
    print(f"✓ Total chunks available: {collection.count()}")
except Exception as e:
    print(f"\n Error: Collection 'video_chunks' not found!")
    print("Please run 'python src/vector_db.py' first to create the collection.")
    exit()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("✓ Groq client ready\n")

def search_chunks(query, top_k=5):
    query_embedding = model.encode([query])[0].tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k 
    )
    
    return results 

def get_answer(query, top_k=5):
    print(f"\nSearching: {query}")
    
    results = search_chunks(query, top_k)
    
    if not results['documents'][0]:
        print("No results found")
        return 
    
    print(f"Found {len(results['documents'][0])} relevant chunks\n")
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"[{i+1}] Index: {meta['index']}, Chunk: {meta['chunk_num']}")
        print(f"    {doc[:100]}...\n")
    
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
- If the exact term isn't defined in the video but related concepts are mentioned, explain the concept using both the video context and general knowledge in less manner.
- Never say "the video doesn't cover this" - instead, provide relevant information from the video
- Explain in topic three to four lines and use bullet points if needed to make it easier to read 
- for all the questions , try to use text from the videos as much as possible 
- Give me answer only one time and do not repeat the answer in different ways 
- it should be handle all type of questions and give me similar answers for all asked answer.
- if the video text is not present so explain the concept in precise 
- it should be handle all type of questions and give me similar answers for all asked answer.

Provide a clear and helpful answer:"""

    print("Generating answer...\n")
    
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000
    )
    
    answer = response.choices[0].message.content
    
    print("=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(answer)
    print("=" * 80)
    
    return answer

print("=" * 80)
print("RAG QUERY SYSTEM - Ready to answer your questions!")
print("=" * 80)

while True:
    question = input("\n Your question (or 'quit' to exit): ").strip()
    
    if question.lower() in ['quit', 'exit', 'q']:
        print("\nBye!")
        break
    
    if not question:
        continue
    
    try:
        get_answer(question, top_k=5)
    except Exception as e:
        print(f"Error: {e}")
        
        