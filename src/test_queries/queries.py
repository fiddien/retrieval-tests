"""Test queries for evaluating LLM models"""


TEST_QUERIES = [
    {
        "id": "cahyono_summary",
        "mode": "summary",
        "query": "Give a full summary of FX Cahyono Wibowo",
        "description": "Full personnel summary for FX Cahyono Wibowo"
    },
    {
        "id": "ardi_summary",
        "mode": "summary",
        "query": "Give a full summary of Ardi Adrian Pualam Sakti",
        "description": "Full personnel summary for Ardi Adrian Pualam Sakti"
    },
    {
        "id": "edi_summary",
        "mode": "summary",
        "query": "Give a full summary of Edi Suyono",
        "description": "Full personnel summary for Edi Suyono"
    },
    {
        "id": "amin_summary",
        "mode": "summary",
        "query": "Give me a full summary of Amin Rahardjo",
        "description": "Full personnel summary for Amin Rahardjo"
    },
    {
        "id": "rudito_summary",
        "mode": "summary",
        "query": "Give me a full summary of Rudito Dwi Wibowo",
        "description": "Full personnel summary for Rudito Dwi Wibowo"
    },
    {
        "id": "cahyono_id_summary",
        "mode": "summary",
        "query": "Tolong berikan informasi lengkap mengenai FX Cahyono Wibowo",
        "description": "Full personnel summary in Indonesian for FX Cahyono Wibowo"
    },
    {
        "id": "ardi_id_summary",
        "mode": "summary",
        "query": "Jelaskan secara detail tentang Ardi Adrian Pualam Sakti",
        "description": "Detailed personnel summary in Indonesian for Ardi Adrian"
    },
    {
        "id": "edi_alt_summary",
        "mode": "summary",
        "query": "What can you tell me about Edi Suyono?",
        "description": "Alternative wording for Edi Suyono summary"
    },
    {
        "id": "amin_id_summary",
        "mode": "summary",
        "query": "Siapa itu Amin Rahardjo? Berikan informasi selengkapnya",
        "description": "Full personnel summary in Indonesian for Amin Rahardjo"
    },
    {
        "id": "rudito_alt_summary",
        "mode": "summary",
        "query": "Could you provide information about Rudito Dwi Wibowo?",
        "description": "Alternative wording for Rudito summary"
    },
    {
        "id": "cahyono_ardi_id_compare",
        "mode": "compare",
        "query": "Bandingkan profil FX Cahyono Wibowo dengan Ardi Adrian Pualam Sakti",
        "description": "Indonesian comparison between Cahyono and Ardi"
    },
    {
        "id": "edi_amin_alt_compare",
        "mode": "compare",
        "query": "What are the differences and similarities between Edi Suyono and Amin Rahardjo?",
        "description": "Alternative wording for comparison between Edi and Amin"
    },
    {
        "id": "rudito_cahyono_id_compare",
        "mode": "compare",
        "query": "Apa perbedaan dan persamaan antara Rudito Dwi Wibowo dengan FX Cahyono Wibowo?",
        "description": "Indonesian comparison between Rudito and Cahyono"
    },
    {
        "id": "ardi_amin_detailed_compare",
        "mode": "compare",
        "query": "Provide a detailed comparison between Ardi Adrian Pualam Sakti and Amin Rahardjo",
        "description": "Detailed comparison request for Ardi and Amin"
    },
    {
        "id": "edi_rudito_id_compare",
        "mode": "compare",
        "query": "Mohon jelaskan perbandingan antara Edi Suyono dengan Rudito Dwi Wibowo",
        "description": "Polite Indonesian comparison between Edi and Rudito"
    },
    {
        "id": "three_way_compare",
        "mode": "compare",
        "query": "Bandingkan pengalaman dan keahlian dari tiga orang berikut: FX Cahyono Wibowo, Amin Rahardjo, dan Rudito Dwi Wibowo",
        "description": "Three-way comparison in Indonesian"
    },
    {
        "id": "experience_focused_compare",
        "mode": "compare",
        "query": "Which one has more relevant experience in their field: Edi Suyono or Ardi Adrian Pualam Sakti? Please focus on their professional background",
        "description": "Experience-focused comparison between Edi and Ardi"
    },
    {
        "id": "leadership_compare",
        "mode": "compare",
        "query": "Dari segi kepemimpinan dan manajemen, siapa yang memiliki track record lebih baik: Amin Rahardjo atau FX Cahyono Wibowo?",
        "description": "Leadership-focused comparison in Indonesian"
    },
    {
        "id": "technical_skills_compare",
        "mode": "compare",
        "query": "I need to understand the technical expertise difference between Rudito Dwi Wibowo and Ardi Adrian Pualam Sakti. Can you analyze their technical backgrounds?",
        "description": "Technical skills focused comparison"
    },
    {
        "id": "career_progression_compare",
        "mode": "compare",
        "query": "Tolong analisa dan bandingkan perkembangan karir dari Edi Suyono dan FX Cahyono Wibowo dari awal hingga posisi terakhir mereka",
        "description": "Career progression comparison in Indonesian"
    }
]
