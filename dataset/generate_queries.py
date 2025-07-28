#!/usr/bin/env python3

import json
import uuid
import random
from collections import defaultdict

def generate_queries():
    # Load corpus
    corpus_entries = []
    with open("dataset/corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            corpus_entries.append(json.loads(line))

    queries = []
    qrels = defaultdict(dict)  # query_id -> {doc_id: relevance}

    # Generate different types of queries based on the data
    query_templates = {
        "name_search": [
            "Siapa yang bernama {name}?",
            "Cari informasi tentang {name}",
            "Data pribadi {name}"
        ],
        "award_search": [
            "Siapa yang mendapat penghargaan {award}?",
            "Daftar penerima {award}",
            "Informasi penghargaan {award}"
        ],
        "position_search": [
            "Siapa yang menjabat sebagai {position}?",
            "Daftar {position}",
            "Informasi jabatan {position}"
        ],
        "date_search": [
            "Siapa yang mendapat penghargaan pada tanggal {date}?",
            "Daftar penghargaan bulan {month} tahun {year}",
            "Penghargaan pada {date}"
        ],
        "location_search": [
            "Siapa yang tinggal di {location}?",
            "Daftar penduduk {location}",
            "Informasi dari {location}"
        ]
    }

    sampled_entries = random.sample(corpus_entries, min(50, len(corpus_entries)))

    for entry in sampled_entries:
        parsed_data = entry["metadata"]["parsed_data"]
        doc_id = entry["_id"]

        # Generate name-based queries
        if "full_name/nama" in parsed_data and parsed_data["full_name/nama"]:
            name = parsed_data["full_name/nama"]
            query_text = random.choice(query_templates["name_search"]).format(name=name)
            query_id = str(uuid.uuid4())

            queries.append({
                "_id": query_id,
                "text": query_text,
                "metadata": {
                    "query_type": "name_search",
                    "target_name": name
                }
            })
            qrels[query_id][doc_id] = 2  # High relevance

        # Generate award-based queries
        if "award_name/nama_penghargaan" in parsed_data and parsed_data["award_name/nama_penghargaan"]:
            award = parsed_data["award_name/nama_penghargaan"]
            query_text = random.choice(query_templates["award_search"]).format(award=award)
            query_id = str(uuid.uuid4())

            queries.append({
                "_id": query_id,
                "text": query_text,
                "metadata": {
                    "query_type": "award_search",
                    "target_award": award
                }
            })
            qrels[query_id][doc_id] = 2

            # Find other docs with same award (lower relevance)
            for other_entry in corpus_entries:
                if other_entry["_id"] != doc_id:
                    other_parsed = other_entry["metadata"]["parsed_data"]
                    if ("award_name/nama_penghargaan" in other_parsed and
                        other_parsed["award_name/nama_penghargaan"] == award):
                        qrels[query_id][other_entry["_id"]] = 1

        # Generate position-based queries
        if "current_position/jabatan_baru" in parsed_data and parsed_data["current_position/jabatan_baru"]:
            position = parsed_data["current_position/jabatan_baru"]
            query_text = random.choice(query_templates["position_search"]).format(position=position)
            query_id = str(uuid.uuid4())

            queries.append({
                "_id": query_id,
                "text": query_text,
                "metadata": {
                    "query_type": "position_search",
                    "target_position": position
                }
            })
            qrels[query_id][doc_id] = 2

    # Write queries.jsonl
    with open("dataset/queries.jsonl", 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')

    # Split qrels into train/dev/test
    query_ids = list(qrels.keys())
    random.shuffle(query_ids)

    train_size = int(0.7 * len(query_ids))
    dev_size = int(0.15 * len(query_ids))

    train_queries = query_ids[:train_size]
    dev_queries = query_ids[train_size:train_size + dev_size]
    test_queries = query_ids[train_size + dev_size:]

    # Write qrels files
    def write_qrels(filename, query_list):
        with open(f"dataset/qrels/{filename}", 'w') as f:
            for qid in query_list:
                for doc_id, relevance in qrels[qid].items():
                    f.write(f"{qid}\t0\t{doc_id}\t{relevance}\n")

    write_qrels("train.tsv", train_queries)
    write_qrels("dev.tsv", dev_queries)
    write_qrels("test.tsv", test_queries)

    print(f"Generated {len(queries)} queries")
    print(f"Train: {len(train_queries)} queries")
    print(f"Dev: {len(dev_queries)} queries")
    print(f"Test: {len(test_queries)} queries")

if __name__ == "__main__":
    random.seed(42)
    generate_queries()