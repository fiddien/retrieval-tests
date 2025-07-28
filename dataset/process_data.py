#!/usr/bin/env python3

import json
import os
import uuid
from pathlib import Path

def process_txt_files():
    raw_data_dir = Path("raw_data")
    corpus_entries = []
    
    # Walk through all txt files
    for txt_file in raw_data_dir.rglob("*.txt"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Parse JSON content
            json_data = json.loads(content)
            
            # Create corpus entry
            corpus_id = str(uuid.uuid4())
            
            # Extract document type from path
            doc_type = txt_file.parent.name
            
            corpus_entry = {
                "_id": corpus_id,
                "text": content,
                "metadata": {
                    "document_type": doc_type,
                    "file_path": str(txt_file),
                    "parsed_data": json_data
                }
            }
            
            corpus_entries.append(corpus_entry)
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Error processing {txt_file}: {e}")
            continue
    
    # Write corpus.jsonl
    with open("dataset/corpus.jsonl", 'w', encoding='utf-8') as f:
        for entry in corpus_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(corpus_entries)} documents into corpus.jsonl")
    return corpus_entries

if __name__ == "__main__":
    corpus_entries = process_txt_files()