import pandas as pd  
from sentence_transformers import SentenceTransformer 


df = pd.read_csv('Data\\processed\\chunks.csv')
chunk_text = df['text'].astype(str).tolist()

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunk_text , show_progress_bar=True , batch_size=32)

print("Generated embeddings for the chunks of text.")

print("Embeddings shape:", embeddings.shape)


df['embeddings'] = embeddings.tolist() 
df.to_csv("Data\embeddings\chunk_embeddings.csv", index=False)
print("Saved the embeddings to chunk_embeddings.csv")
