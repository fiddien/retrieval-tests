Goal: make a BEIR-formatted retrieval task dataset, given txt files of the kind of this JSON-formatted string called chunk:

```
{
  "full_name/nama": "RUDITO DWI WIBOWO",
  "award_name/nama_penghargaan": "SATYALANCANA DWIDYA SISTHA",
  "award_date/tanggal_penghargaan": "06-10-2011"
}
```

The structure of this dataset should be like this:
```
.
├── README.md
└── dataset
    ├── corpus.jsonl
    ├── metadata.json
    ├── qrels
    │   ├── dev.tsv
    │   ├── test.tsv
    │   └── train.tsv
    └── queries.jsonl
```

each corpus json contains at least something like:
```
{"_id": "c7d66b69-9c05-496d-9182-aee84d6d331a", "text": "chunk content string in here", "metadata": {...}}
```

each query json contains at least something akin to this format:

```
{"_id": "fa51e425-3ae4-4a91-a108-a478131a8bc1", "text": "Mengapa pemungutan suara serentak pada bulan September tahun 2020 tidak dapat dilaksanakan sesuai jadwal yang telah ditentukan?", "metadata": {...}}
```
