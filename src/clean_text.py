import re
import csv 

def convert_transcribe_proper_format(input_file, output_file):
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = re.sub(r'<-+Video\d+-+>', '', content)
    content = re.sub(r'\(\d+:\d+\)', '', content)
    content = re.sub(r'\d+:\d+', '', content)
    
    lines = content.strip().split('\n')
    
    all_text_row = []
    
    for line in lines:
        cleaned_line = line.strip()
        if cleaned_line:
            all_text_row.append({'text': cleaned_line})
            
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text'])
        writer.writeheader()
        writer.writerows(all_text_row)
    
    print(f"converted {len(all_text_row)} rows")
    print(f"saved to: {output_file}") 
        
        
if __name__ == "__main__":
    INPUT_FILE = 'Data\\raw\\combine_text.csv'
    OUTPUT_FILE = "input_text.csv"
    
    print("Converting transcript to clean text only...")
    convert_transcribe_proper_format(INPUT_FILE, OUTPUT_FILE)
    print("Done!")