# Agentic Legal RAG Chatbot (DZ)

This repository contains the Legal-only RAG assistant with web search fallback and user document Q&A. It is derived from the main Multilingual RAG Chatbot for the Arabic NLP Platform.

## Project Overview

This system presents the design, implementation, and evaluation of a multilingual Retrieval-Augmented Generation (RAG) chatbot. This specific repository focuses on the **Legal Advisory** mode, supporting Arabic, French, and English. 

The architecture combines a 12-stage processing pipeline—from language detection through faithfulness verification—with a hybrid retrieval engine that fuses dense semantic search (BGE-M3 embeddings, Qdrant) and sparse lexical search (BM25) using Reciprocal Rank Fusion (RRF).

### Key Features
- **Theoretical Framework:** Uses transformer embeddings, BM25 scoring, RRF fusion, and BERTScore evaluation.
- **Knowledge-Base Ingestion:** Uses Docling for structure-aware PDF/XML parsing of academic and legal documents.
- **LLM Provider:** Uses Google Gemini as the primary LLM provider to eliminate hallucination issues in the legal domain.
- **Dual-Channel Web Search:** Incorporates Exa for automatic fallback, and Tavily for user-triggered search.

### Evaluation
Automated evaluation demonstrates strong retrieval performance on the primary benchmark run: 
- **Precision@5:** 0.7524
- **Recall@5:** 0.9405
- **MRR:** 0.8786
- **BERTScore F1:** 0.9472
- **Failures:** 0

## How to Run It

### Prerequisites
- Docker and Docker Compose installed.
- Required API keys (Gemini/Groq, Exa, Tavily).

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/D3epX/Agentic-Legal-RAG-DZ.git
   cd Agentic-Legal-RAG-DZ
   ```

2. **Configure Environment Variables:**
   Copy the example environment file and fill in your keys.
   ```bash
   cp .env.example .env
   ```
   *Make sure to fill in your `GENAI_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY`, and PostgreSQL credentials in `.env`.*

3. **Build and Run with Docker Compose:**
   ```bash
   docker compose up --build -d
   ```

4. **Access the Chatbot:**
   - The Chat UI will be available at: `http://localhost:8000`
   - API Health Check: `http://localhost:8000/health`

## Repository Notes
- The UI has been updated to be legal-only and exposes web search.
- Platform search and NLP/AI mode have been intentionally removed from this interface to match the focused legal chatbot logic.
