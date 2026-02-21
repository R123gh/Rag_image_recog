import pandas as pd  
from langchain_text_splitters import RecursiveCharacterTextSplitter


df = pd.read_csv('Input_data\\input_text.csv')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
)

all_chunks = []

for index, row in df.iterrows():
    text = str(row['text'])  
    
    for chunk_num, chunk in enumerate(splitter.split_text(text)):
        all_chunks.append({
            "index": index,
            "chunk_num": chunk_num,
            "text": chunk
        })
        
pd.DataFrame(all_chunks).to_csv('Data\\processed\\chunks.csv', index=False)
print(f"✓ Created {len(all_chunks)} chunks")