"""
Minimal knowledge base for retrieval-augmented generation (RAG).

This is intentionally small and curated by hand -- the point of this stretch goal
is to demonstrate the RAG *mechanism* (retrieve -> prepend -> generate), not to build
a production-scale document store. Each entry is a short factual snippet keyed by
topic keywords used for retrieval matching.
"""

KNOWLEDGE_BASE = [
    {
        "keywords": ["bangalore", "bengaluru"],
        "text": (
            "Bengaluru (also known as Bangalore) is the capital of the Indian state "
            "of Karnataka. It is known as India's IT hub / Silicon Valley, home to "
            "major technology companies and startups. It has a temperate climate due "
            "to its elevation on the Deccan Plateau."
        ),
    },
    {
        "keywords": ["tirupati", "tirumala"],
        "text": (
            "Tirupati is a city in the Chittoor district of Andhra Pradesh, India. "
            "It is home to the Tirumala Venkateswara Temple, one of the most visited "
            "religious pilgrimage sites in the world."
        ),
    },
    {
        "keywords": ["france", "paris"],
        "text": "Paris is the capital and largest city of France.",
    },
    {
        "keywords": ["japan", "tokyo"],
        "text": "Tokyo is the capital of Japan.",
    },
    {
        "keywords": ["photosynthesis"],
        "text": (
            "Photosynthesis is the process by which plants, algae, and some bacteria "
            "convert light energy, usually from the sun, into chemical energy. Using "
            "sunlight, water, and carbon dioxide, chlorophyll in plant cells produces "
            "glucose (a sugar used for energy) and releases oxygen as a byproduct."
        ),
    },
    {
        "keywords": ["india"],
        "text": (
            "India is a country in South Asia. It is the world's most populous "
            "country and the seventh-largest by land area. New Delhi is its capital."
        ),
    },
]


def retrieve(query: str, min_score: int = 1):
    """
    Very simple keyword-overlap retrieval: lowercase the query, check how many of
    each knowledge-base entry's keywords appear in it, return the best-matching
    entry's text if it clears min_score, else None.

    This is deliberately simple (no embeddings/vector search) so the mechanism is
    easy to explain and verify -- swapping in a real embedding-based retriever
    (e.g. sentence-transformers + cosine similarity) would be the natural next step
    with more time, without changing anything downstream in generate.py.
    """
    query_lower = query.lower()
    best_entry = None
    best_score = 0

    for entry in KNOWLEDGE_BASE:
        score = sum(1 for kw in entry["keywords"] if kw in query_lower)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry and best_score >= min_score:
        return best_entry["text"]
    return None
