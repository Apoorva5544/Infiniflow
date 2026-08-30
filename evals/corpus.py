"""
Offline evaluation corpus for Infiniflow.

Deterministic, dependency-light fixture: a set of (source, page, text) chunks
covering several topics, plus a set of questions with ground-truth answers and
the sources that should be retrieved for each.

Used by `evals/evaluate.py` to compute retrieval metrics (context precision@k)
and — when a GROQ_API_KEY is present — generation metrics (faithfulness).
"""

CORPUS = [
    {
        "source": "europe_guide.pdf",
        "page": 1,
        "text": (
            "Paris is the capital of France and is located on the River Seine. "
            "It is home to the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
        ),
    },
    {
        "source": "europe_guide.pdf",
        "page": 2,
        "text": (
            "France is a founding member of the European Union and the United Nations. "
            "The country is governed as a republic with a president and a prime minister."
        ),
    },
    {
        "source": "space_primer.pdf",
        "page": 1,
        "text": (
            "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. "
            "It was completed in 1889 as the entrance arch to the World's Fair."
        ),
    },
    {
        "source": "space_primer.pdf",
        "page": 3,
        "text": (
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn from data without being explicitly programmed. "
            "Neural networks are one popular class of machine learning model."
        ),
    },
    {
        "source": "company_handbook.pdf",
        "page": 5,
        "text": (
            "Infiniflow offers a free tier that includes 1,000 queries per month "
            "and hybrids search across up to 3 workspaces. Paid plans add "
            "cross-encoder reranking and unlimited streaming with citations."
        ),
    },
    {
        "source": "company_handbook.pdf",
        "page": 6,
        "text": (
            "Employee onboarding takes two weeks and includes security training, "
            "a laptop handover, and a guided tour of our internal RAG platform."
        ),
    },
    {
        "source": "climate_report.pdf",
        "page": 2,
        "text": (
            "Global average temperature has risen approximately 1.2 degrees Celsius "
            "since pre-industrial times, driven overwhelmingly by fossil fuel emissions."
        ),
    },
    {
        "source": "climate_report.pdf",
        "page": 4,
        "text": (
            "Renewable energy sources such as solar and wind generated roughly 30% "
            "of global electricity in 2023 and are the fastest-growing energy sources."
        ),
    },
]

QUESTIONS = [
    {
        "question": "What is the capital of France and which famous monuments are there?",
        "ground_truth": "The capital of France is Paris, located on the Seine, with the Eiffel Tower, Louvre Museum, and Notre-Dame.",
        "expected_sources": ["europe_guide.pdf", "space_primer.pdf"],
    },
    {
        "question": "What is machine learning and what are neural networks?",
        "ground_truth": "Machine learning is a subset of AI where systems learn from data; neural networks are a class of ML model.",
        "expected_sources": ["space_primer.pdf"],
    },
    {
        "question": "How much has global temperature risen and what causes it?",
        "ground_truth": "Global temperatures rose ~1.2C since pre-industrial times, driven by fossil fuel emissions.",
        "expected_sources": ["climate_report.pdf"],
    },
    {
        "question": "What does the Infiniflow free tier include?",
        "ground_truth": "Free tier includes 1,000 queries per month and hybrid search across up to 3 workspaces.",
        "expected_sources": ["company_handbook.pdf"],
    },
]
