import os
import re
import logging
from pathlib import Path
from typing import TypedDict, List, Any

import fitz
import numpy as np
import faiss
import gradio as gr
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consumer-care-ai")

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "documents"))

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
NEXUS_BASE_URL = "https://nexusapi.navigatelabs.ai/v1"
NEXUS_MODEL_NAME = "nova-micro"

NEXUS_API_KEY = os.getenv("NEXUS_API_KEY")
if not NEXUS_API_KEY:
    raise RuntimeError(
        "NEXUS_API_KEY is missing. Add it as an environment variable "
        "in your hosting provider's dashboard. Do not put the key in this file."
    )

client = OpenAI(api_key=NEXUS_API_KEY, base_url=NEXUS_BASE_URL)

# ------------------------------------------------------------
# LOAD OFFICIAL PDF DOCUMENTS AND BUILD KNOWLEDGE BASE
# ------------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 150):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def load_pdf_chunks():
    if not DOCUMENTS_DIR.exists():
        raise RuntimeError(
            f"PDF folder not found: {DOCUMENTS_DIR}. "
            "Create a 'documents' folder in the repository and put the "
            "consumer-protection PDF files inside it."
        )

    pdf_paths = sorted(DOCUMENTS_DIR.rglob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(
            f"No PDF files found in {DOCUMENTS_DIR}. "
            "Upload the consumer-protection PDFs into the documents folder."
        )

    records = []
    for pdf_path in pdf_paths:
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_text = clean_text(page.get_text("text"))
                    if not page_text:
                        continue

                    for chunk_id, chunk in enumerate(create_chunks(page_text)):
                        records.append({
                            "document_name": pdf_path.name,
                            "page_number": page_number,
                            "chunk_id": chunk_id,
                            "text": chunk,
                        })
        except Exception:
            logger.exception("Could not read PDF: %s", pdf_path.name)

    if not records:
        raise RuntimeError("The PDF files were found, but no readable text was extracted.")

    logger.info("Loaded %s PDF files and created %s text chunks.", len(pdf_paths), len(records))
    return records


logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

logger.info("Loading Cross-Encoder model: %s", RERANKER_MODEL_NAME)
reranker = CrossEncoder(RERANKER_MODEL_NAME)

chunks = load_pdf_chunks()
chunk_texts = [item["text"] for item in chunks]

logger.info("Creating document embeddings...")
embeddings = embedding_model.encode(
    chunk_texts,
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True,
).astype("float32")

vector_index = faiss.IndexFlatIP(embeddings.shape[1])
vector_index.add(embeddings)

def tokenize(text):
    return re.findall(r"\w+", str(text).lower(), flags=re.UNICODE)

tokenized_corpus = [tokenize(item["text"]) for item in chunks]
bm25 = BM25Okapi(tokenized_corpus)

logger.info("Knowledge base ready: %s chunks, %s FAISS vectors.", len(chunks), vector_index.ntotal)

# ------------------------------------------------------------
# QUERY EXPANSION
# ------------------------------------------------------------

def expand_query(query):
    q = query.lower()
    variations = []

    if "consumer" in q and ("right" in q or "protection" in q):
        variations = [
            "What are the six consumer rights under Indian law?",
            "Right to safety and protection against hazardous goods",
            "Right to be informed about quality quantity and price",
            "Right to choose goods and services",
            "Right to be heard and seek consumer redressal",
            "Right to consumer education",
            "Consumer rights under Consumer Protection Act 2019",
        ]
    elif "refund" in q or "return" in q:
        variations = [
            "Consumer rights regarding refunds",
            "Return and replacement of defective products",
            "Refund policy under consumer protection law",
            "Compensation for defective goods",
            "Unfair trade practices related to refunds",
        ]
    elif "online shopping" in q or "e-commerce" in q:
        variations = [
            "E-commerce consumer protection rules",
            "Online shopping complaints and refunds",
            "Responsibilities of e-commerce platforms",
            "Consumer rights against online sellers",
        ]
    elif "complaint" in q or "grievance" in q:
        variations = [
            "Consumer complaint filing procedure",
            "Consumer dispute redressal process",
            "District Consumer Commission complaint",
            "Consumer grievance resolution",
        ]

    return list(dict.fromkeys([query] + variations))


# ------------------------------------------------------------
# HYBRID RETRIEVAL: FAISS + BM25 + RECIPROCAL RANK FUSION
# ------------------------------------------------------------

def hybrid_search(query, top_k=10, candidate_k=20, rrf_k=60):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    _, dense_indices = vector_index.search(
        query_embedding, min(candidate_k, len(chunks))
    )

    sparse_scores = bm25.get_scores(tokenize(query))
    sparse_indices = np.argsort(sparse_scores)[::-1][:candidate_k]

    rrf_scores = {}
    for rank, idx in enumerate(dense_indices[0], start=1):
        if idx >= 0:
            idx = int(idx)
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)

    for rank, idx in enumerate(sparse_indices, start=1):
        idx = int(idx)
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_k + rank)

    ranked_indices = sorted(
        rrf_scores, key=rrf_scores.get, reverse=True
    )[:top_k]

    return [
        {
            "document_name": chunks[idx]["document_name"],
            "page_number": chunks[idx]["page_number"],
            "text": chunks[idx]["text"],
            "rrf_score": rrf_scores[idx],
        }
        for idx in ranked_indices
    ]


def expanded_hybrid_search(query, top_k=10):
    combined_scores = {}
    result_data = {}

    for expanded_query in expand_query(query):
        results = hybrid_search(expanded_query, top_k=top_k)
        for rank, result in enumerate(results, start=1):
            key = (
                result["document_name"],
                result["page_number"],
                result["text"],
            )
            combined_scores[key] = combined_scores.get(key, 0) + 1 / (60 + rank)
            result_data[key] = result

    ranked_keys = sorted(
        combined_scores, key=combined_scores.get, reverse=True
    )[:top_k]

    final_results = []
    for key in ranked_keys:
        item = result_data[key].copy()
        item["expanded_score"] = combined_scores[key]
        final_results.append(item)

    return final_results


# ------------------------------------------------------------
# CROSS-ENCODER RERANKING AND LEGAL-AWARE RETRIEVAL
# ------------------------------------------------------------

def cross_encoder_rerank(query, results, top_k=5):
    if not results:
        return []

    pairs = [[query, result["text"]] for result in results]
    scores = reranker.predict(pairs)

    reranked = []
    for result, score in zip(results, scores):
        item = result.copy()
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]


def consumer_retrieval_v2(question, candidate_k=20, final_k=5):
    question_lower = question.lower()

    remedy_keywords = [
        "refund", "money back", "return my money",
        "defective", "damaged product", "replacement",
        "replace the product", "faulty product",
    ]
    is_remedy_query = any(word in question_lower for word in remedy_keywords)

    normal_results = expanded_hybrid_search(question, top_k=candidate_k)
    all_results = normal_results.copy()

    if is_remedy_query:
        legal_query = (
            "Section 39 Consumer Protection Act 2019 remedies "
            "replacement of defective goods return to complainant "
            "price or charges paid compensation for loss or injury"
        )
        legal_results = hybrid_search(legal_query, top_k=candidate_k)
        all_results.extend(legal_results)
        rerank_query = legal_query
    else:
        rerank_query = question

    unique_results = {}
    for item in all_results:
        key = (item["document_name"], item["page_number"], item["text"])
        unique_results[key] = item

    return cross_encoder_rerank(
        rerank_query, list(unique_results.values()), top_k=final_k
    )


# ------------------------------------------------------------
# RAG PROMPT AND NEXUS RESPONSE GENERATION
# ------------------------------------------------------------

def build_context(results):
    context = ""
    for i, result in enumerate(results, 1):
        context += (
            f"\nSource {i}\n"
            f"Document: {result['document_name']}\n"
            f"Page: {result['page_number']}\n"
            f"Content:\n{result['text']}\n"
            "----------------------------------------\n"
        )
    return context


def create_rag_prompt(query, context):
    return f"""
You are an AI Consumer Complaint Resolution Assistant.

Answer the user's question using ONLY the information
provided in the official document context below.

Instructions:
1. Give clear and simple answers.
2. Do not invent legal information.
3. If the context does not contain the answer, say:
   "The available documents do not provide sufficient information."
4. Include document names and page numbers as sources.
5. Use bullet points when explaining multiple items.

OFFICIAL DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""


def generate_rag_answer(prompt):
    response = client.chat.completions.create(
        model=NEXUS_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or "No response was returned."


# ------------------------------------------------------------
# FIVE LANGGRAPH AGENTS + IN-MEMORY CONTEXT CACHE
# ------------------------------------------------------------

class ConsumerAgentState(TypedDict):
    question: str
    retrieved_results: List[Any]
    reranked_results: List[Any]
    context: str
    is_relevant: bool
    final_answer: str


context_cache = {}
cache_status = "NOT TESTED"


def query_analysis_agent(state: ConsumerAgentState):
    return {"question": state["question"].strip()}


def retrieval_agent(state: ConsumerAgentState):
    global cache_status

    question = state["question"].strip()
    cache_key = question.lower()

    if cache_key in context_cache:
        cache_status = "HIT"
        cached_data = context_cache[cache_key]
        return {
            "retrieved_results": cached_data["retrieved_results"],
            "reranked_results": cached_data["reranked_results"],
            "context": cached_data["context"],
        }

    cache_status = "MISS"
    reranked_results = consumer_retrieval_v2(
        question, candidate_k=20, final_k=5
    )
    return {
        "retrieved_results": reranked_results,
        "reranked_results": reranked_results,
    }


def relevance_validation_agent(state: ConsumerAgentState):
    results = state["reranked_results"]
    if not results:
        relevant = False
    else:
        # Preserves the notebook's relevance threshold.
        relevant = max(item["rerank_score"] for item in results) >= 0.0
    return {"is_relevant": relevant}


def context_building_agent(state: ConsumerAgentState):
    if state.get("context"):
        context = state["context"]
    elif state["is_relevant"]:
        context = build_context(state["reranked_results"])
    else:
        context = ""

    cache_key = state["question"].strip().lower()
    context_cache[cache_key] = {
        "retrieved_results": state["retrieved_results"],
        "reranked_results": state["reranked_results"],
        "context": context,
    }
    return {"context": context}


def response_generation_agent(state: ConsumerAgentState):
    if not state["is_relevant"]:
        answer = (
            "Insufficient relevant information was found in the "
            "available consumer-protection documents."
        )
    else:
        prompt = create_rag_prompt(state["question"], state["context"])
        answer = generate_rag_answer(prompt)
    return {"final_answer": answer}


workflow = StateGraph(ConsumerAgentState)
workflow.add_node("query_analysis", query_analysis_agent)
workflow.add_node("retrieval", retrieval_agent)
workflow.add_node("validation", relevance_validation_agent)
workflow.add_node("context_building", context_building_agent)
workflow.add_node("response_generation", response_generation_agent)

workflow.add_edge(START, "query_analysis")
workflow.add_edge("query_analysis", "retrieval")
workflow.add_edge("retrieval", "validation")
workflow.add_edge("validation", "context_building")
workflow.add_edge("context_building", "response_generation")
workflow.add_edge("response_generation", END)

consumer_agent = workflow.compile()


def consumer_chatbot(user_question):
    if not user_question or not user_question.strip():
        return "Please enter a consumer-related question."

    initial_state = {
        "question": user_question.strip(),
        "retrieved_results": [],
        "reranked_results": [],
        "context": "",
        "is_relevant": False,
        "final_answer": "",
    }
    result = consumer_agent.invoke(initial_state)
    return result["final_answer"]


# ------------------------------------------------------------
# GRADIO WEBSITE
# ------------------------------------------------------------

def respond(message, history):
    if not message or not message.strip():
        return history or [], ""

    answer = consumer_chatbot(message.strip())
    updated_history = (history or []) + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, ""


custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.hero {
    padding: 35px;
    border-radius: 16px;
    background: linear-gradient(120deg, #163b65, #267c91);
    color: white;
    text-align: center;
}
.hero h1 { font-size: 34px; }
"""

with gr.Blocks(title="Consumer Care AI", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
    <div class="hero">
        <h1>Consumer Care AI</h1>
        <h3>Agentic RAG-Based Consumer Complaint Resolution Assistant</h3>
        <p>Understand your consumer rights and get guidance
        from official Indian consumer-protection documents.</p>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Home"):
            gr.Markdown("""
## Welcome to Consumer Care AI

Consumer Care AI helps people understand consumer rights and complaint
resolution procedures in India.

### Key Features
- Retrieval from official government documents
- FAISS semantic search and BM25 keyword search
- Query expansion and Cross-Encoder reranking
- Five task-specialized LangGraph agents
- Context caching for repeated questions
- Official document and page references

Ask a consumer-related question in the AI Chatbot tab.
""")

        with gr.Tab("Consumer Rights"):
            gr.Markdown("""
## Six Consumer Rights in India

1. **Right to Safety** – Protection against hazardous goods and services.
2. **Right to be Informed** – Information about quality, quantity, purity, standard and price.
3. **Right to Choose** – Access to a variety of goods and services at competitive prices.
4. **Right to be Heard** – Consumer interests receive consideration at appropriate forums.
5. **Right to Seek Redressal** – Protection against unfair trade practices and exploitation.
6. **Right to Consumer Education** – The right to consumer awareness.

**Reference in the project knowledge base:** Consumer Protection Act, 2019.
""")

        with gr.Tab("Complaint Guidance"):
            gr.Markdown("""
## Consumer Complaint Guidance

### If you receive a defective product
- Keep the purchase receipt and order details.
- Preserve photographs or other evidence of the defect.
- Contact the seller or service provider.
- You may seek appropriate remedies through the consumer dispute redressal system.

### If you want a refund
- Keep proof of payment and purchase.
- Review the seller's applicable return and refund terms.
- Where applicable, a Consumer Commission may order remedies under Section 39
  of the Consumer Protection Act, 2019.

### Official Consumer Resources
- [National Consumer Helpline](https://consumerhelpline.gov.in/)
- [Department of Consumer Affairs](https://consumeraffairs.nic.in/)
- [India Code](https://www.indiacode.nic.in/)

*This assistant provides information, not legal advice.*
""")

        with gr.Tab("AI Chatbot"):
            gr.Markdown("""
## Ask Consumer Care AI

Ask about consumer rights, defective products, refunds and complaint resolution.
Answers are generated using the official documents in the knowledge base.
""")
            chatbot = gr.Chatbot(type="messages", height=430)
            with gr.Row():
                message = gr.Textbox(
                    placeholder="Example: How can I seek a refund?",
                    show_label=False,
                    lines=2,
                    scale=8,
                )
                send_button = gr.Button("Send", variant="primary", scale=1)

            clear_button = gr.Button("Clear Chat")
            message.submit(respond, inputs=[message, chatbot], outputs=[chatbot, message])
            send_button.click(respond, inputs=[message, chatbot], outputs=[chatbot, message])
            clear_button.click(lambda: ([], ""), outputs=[chatbot, message])

    gr.Markdown("""
---
**Consumer Care AI | AI Agents + RAG + Context Caching**

Developed as an academic capstone prototype.
""")

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
