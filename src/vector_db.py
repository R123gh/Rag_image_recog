import pandas as pd
import chromadb

df = pd.read_csv("Data\embeddings\chunk_embeddings.csv")

chunks_text = df["text"].astype(str).tolist()
embeddings = df["embeddings"].apply(eval).tolist()

client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection(name="video_chunks")
    print("✓ Deleted old collection")
except:
    pass

collection = client.get_or_create_collection(
    name="video_chunks"
)

collection.add(
    documents=chunks_text,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks_text))],
    metadatas=[
        {
            "index": int(row["index"]),
            "chunk_num": int(row["chunk_num"])
        }
        for _, row in df.iterrows()
    ]
)

print(f"✓ Stored {len(embeddings)} embeddings in ChromaDB")
print(f"✓ Embedding dimension: {len(embeddings[0])}")
print(f"✓ Collection name: video_chunks")