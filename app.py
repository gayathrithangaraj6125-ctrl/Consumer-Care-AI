import os
import re
import logging
from pathlib import Path
from typing import TypedDict, List, Any

import fitz
import gradio as gr
from rank_bm25 import BM25Okapi
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consumer-care-ai")

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", BASE_DIR / "documents"))
NEXUS_API_KEY = os.getenv("NEXUS_API_KEY")
NEXUS_BASE_URL = "https://nexusapi.navigatelabs.ai/v1"
NEXUS_MODEL_NAME = "nova-micro"

if not NEXUS_API_KEY:
    raise RuntimeError("Missing NEXUS_API_KEY environment variable. Add it in Render's Environment settings.")

client = OpenAI(api_key=NEXUS_API_KEY, base_url=NEXUS_BASE_URL)

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, size: int = 1000, overlap: int = 150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            split_at = text.rfind(" ", start, end)
            if split_at > start:
                end = split_at
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks

def tokenize(text: str):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

def load_documents():
    if not DOCUMENTS_DIR.exists():
        raise RuntimeError(f"Documents folder not found: {DOCUMENTS_DIR}")
    pdf_files = sorted(DOCUMENTS_DIR.rglob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {DOCUMENTS_DIR}")

    records = []
    for path in pdf_files:
        try:
            with fitz.open(path) as pdf:
                for page_no, page in enumerate(pdf, start=1):
                    text = clean_text(page.get_text("text"))
                    for chunk_no, chunk in enumerate(chunk_text(text)):
                        records.append({
                            "document_name": path.name,
                            "page_number": page_no,
                            "chunk_id": chunk_no,
                            "text": chunk
                        })
        except Exception:
            logger.exception("Unable to read %s", path.name)

    if not records:
        raise RuntimeError("PDFs were found, but no readable text was extracted.")
    logger.info("Loaded %d PDFs and %d chunks", len(pdf_files), len(records))
    return records

# Lightweight retrieval: BM25 avoids loading large embedding and reranking models.
chunks = load_documents()
bm25 = BM25Okapi([tokenize(item["text"]) for item in chunks])

def expand_query(query: str):
    q = query.lower()
    extra = []
    if "refund" in q or "return" in q:
        extra = ["consumer refund remedy", "return defective goods", "replacement or refund under consumer law"]
    elif "complaint" in q or "grievance" in q:
        extra = ["consumer complaint filing procedure", "consumer dispute redressal commission"]
    elif "right" in q or "consumer protection" in q:
        extra = ["six consumer rights under Consumer Protection Act 2019", "right to safety information choice redressal consumer education"]
    return list(dict.fromkeys([query] + extra))

def retrieve(question: str, top_k: int = 5):
    # Merge BM25 rankings from the original and expanded questions.
    scores = {}
    for q in expand_query(question):
        raw = bm25.get_scores(tokenize(q))
        for rank, idx in enumerate(sorted(range(len(raw)), key=lambda i: raw[i], reverse=True)[:20], start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (60 + rank)
    selected = sorted(scores, key=scores.get, reverse=True)[:top_k]
    results = []
    for idx in selected:
        item = dict(chunks[idx])
        item["retrieval_score"] = scores[idx]
        results.append(item)
    return results

def build_context(results):
    return "\n\n".join(
        f"Source {i}\nDocument: {r['document_name']}\nPage: {r['page_number']}\nContent: {r['text']}"
        for i, r in enumerate(results, 1)
    )

def answer_from_context(question: str, context: str):
    prompt = f"""You are Consumer Care AI, an information assistant about Indian consumer protection.
Answer using ONLY the official document context below. Do not invent legal rules.
If the context does not support an answer, say: "The available documents do not provide sufficient information."
Use simple language and cite document names and page numbers.
This is general information, not legal advice.

OFFICIAL DOCUMENT CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    response = client.chat.completions.create(
        model=NEXUS_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or "No response was returned."

class AgentState(TypedDict):
    question: str
    retrieved_results: List[Any]
    context: str
    is_relevant: bool
    final_answer: str

context_cache = {}

def query_analysis_agent(state: AgentState):
    return {"question": state["question"].strip()}

def retrieval_agent(state: AgentState):
    key = state["question"].strip().lower()
    if key in context_cache:
        cached = context_cache[key]
        return {"retrieved_results": cached["results"], "context": cached["context"]}
    results = retrieve(state["question"])
    return {"retrieved_results": results}

def validation_agent(state: AgentState):
    results = state["retrieved_results"]
    # BM25 returns a ranked list; require at least one non-zero lexical match.
    relevant = bool(results) and results[0].get("retrieval_score", 0) > 0
    return {"is_relevant": relevant}

def context_agent(state: AgentState):
    key = state["question"].strip().lower()
    context = state.get("context") or (build_context(state["retrieved_results"]) if state["is_relevant"] else "")
    context_cache[key] = {"results": state["retrieved_results"], "context": context}
    return {"context": context}

def response_agent(state: AgentState):
    if not state["is_relevant"]:
        answer = "The available documents do not provide sufficient relevant information to answer this question."
    else:
        answer = answer_from_context(state["question"], state["context"])
    return {"final_answer": answer}

graph = StateGraph(AgentState)
graph.add_node("query_analysis", query_analysis_agent)
graph.add_node("retrieval", retrieval_agent)
graph.add_node("validation", validation_agent)
graph.add_node("context_building", context_agent)
graph.add_node("response_generation", response_agent)
graph.add_edge(START, "query_analysis")
graph.add_edge("query_analysis", "retrieval")
graph.add_edge("retrieval", "validation")
graph.add_edge("validation", "context_building")
graph.add_edge("context_building", "response_generation")
graph.add_edge("response_generation", END)
consumer_agent = graph.compile()

def chatbot(message, history):
    if not message or not message.strip():
        return (history or []), ""
    state = {
        "question": message.strip(),
        "retrieved_results": [],
        "context": "",
        "is_relevant": False,
        "final_answer": "",
    }
    result = consumer_agent.invoke(state)
    updated = (history or []) + [
        {"role": "user", "content": message.strip()},
        {"role": "assistant", "content": result["final_answer"]},
    ]
    return updated, ""

CSS = """
.gradio-container { max-width: 1150px !important; margin: auto !important; }
.hero { padding: 30px; border-radius: 16px; background: linear-gradient(120deg,#163b65,#267c91); color: white; text-align: center; }
"""

with gr.Blocks(title="Consumer Care AI", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.HTML("""
    <div class="hero">
      <h1>Consumer Care AI</h1>
      <h3>Agentic RAG-Based Consumer Complaint Resolution Assistant</h3>
      <p>Understand consumer rights and complaint procedures using official Indian documents.</p>
    </div>
    """)
    with gr.Tabs():
        with gr.Tab("Home"):
            gr.Markdown("""
## Welcome to Consumer Care AI
This prototype retrieves information from official consumer-protection PDFs and generates grounded answers with document and page references.

**Features:** BM25 keyword retrieval, query expansion, five LangGraph agents, context caching, and Nexus-powered answer generation.
""")
        with gr.Tab("Consumer Rights"):
            gr.Markdown("""
## Six Consumer Rights in India
1. **Right to Safety** – Protection against hazardous goods and services.
2. **Right to be Informed** – Information about quality, quantity, purity, standard and price.
3. **Right to Choose** – Access to a variety of goods and services at competitive prices.
4. **Right to be Heard** – Consumer interests receive consideration.
5. **Right to Seek Redressal** – Redress against unfair trade practices and exploitation.
6. **Right to Consumer Education** – Consumer awareness.

Refer to the Consumer Protection Act, 2019 in the knowledge base.
""")
        with gr.Tab("Complaint Guidance"):
            gr.Markdown("""
## Complaint Guidance
- Keep receipts, order details, payment proof and relevant photographs.
- Contact the seller or service provider and retain the communication.
- Consult the official consumer resources below for current procedures.

**Official resources**
- [National Consumer Helpline](https://consumerhelpline.gov.in/)
- [Department of Consumer Affairs](https://consumeraffairs.nic.in/)
- [India Code](https://www.indiacode.nic.in/)

*This assistant provides general information, not legal advice.*
""")
        with gr.Tab("AI Chatbot"):
            gr.Markdown("Ask about consumer rights, defective products, refunds and complaint resolution.")
            chat = gr.Chatbot(type="messages", height=430)
            with gr.Row():
                msg = gr.Textbox(placeholder="Example: What can I do if I receive a defective product?", show_label=False, lines=2, scale=8)
                send = gr.Button("Send", variant="primary", scale=1)
            clear = gr.Button("Clear Chat")
            msg.submit(chatbot, [msg, chat], [chat, msg])
            send.click(chatbot, [msg, chat], [chat, msg])
            clear.click(lambda: ([], ""), outputs=[chat, msg])
    gr.Markdown("---\n**Consumer Care AI | AI Agents + RAG + Context Caching**")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=False)
