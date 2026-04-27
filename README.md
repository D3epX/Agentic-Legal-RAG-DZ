# Agentic Legal RAG Chatbot (DZ)

This repository contains the standalone Legal-only RAG assistant with web search fallback and user document Q&A. It is adapted from the main Multilingual RAG Chatbot for the Arabic NLP Platform to serve exclusively as a highly reliable Legal Advisor.

## 📖 Architecture & Theoretical Framework

This system is built upon a state-of-the-art **12-stage processing pipeline** designed to eliminate hallucinations in the high-stakes legal domain. Below are the key components and theoretical concepts that power the chatbot:

### 1. Hybrid Retrieval Engine (Dense + Sparse Search)
The system leverages a **Hybrid Retrieval Engine** to find the most relevant legal context. It combines:
- **Dense Semantic Search:** Uses BGE-M3 (a multilingual embedding model supporting 100+ languages) embedded into a 1024-dimensional vector space. Qdrant handles approximate nearest neighbour (ANN) searches via HNSW indexing.
- **Sparse Lexical Search:** Uses BM25 (Best Matching 25) with a custom multilingual tokeniser to ensure exact-match terms (like specific legal article numbers or proper nouns) are never missed.

### 2. Reciprocal Rank Fusion (RRF) & Passage Reranking
To merge the semantic and lexical results without fragile manual weight tuning, the system utilizes **Reciprocal Rank Fusion (RRF)**. 
- After fusion, a **Jaccard Similarity** check eliminates duplicate chunks.
- Finally, the top candidates undergo a **Cosine Reranking** step to re-score the precise contextual relevance before sending the prompt to the LLM.

### 3. Faithfulness Verification Gate
In the legal domain, hallucinated article numbers or misquoted regulations can lead to incorrect advice. The chatbot implements a **Faithfulness Verification Gate**:
- Before displaying an answer, a secondary LLM strictly classifies the generated response as `faithful` or `not_faithful` against the retrieved legal context.
- If no legal sources are retrieved, or if the answer is deemed unfaithful, the assistant defaults to a safe refusal.

### 4. Dual LLM Provider Architecture
The system employs a primary and fallback LLM architecture:
- **Google Gemini (Primary):** Provides natively strong multilingual reasoning and low hallucination rates in Arabic legal contexts.
- **Groq LLaMA (Fallback):** Ultra-low latency fallback to ensure high availability.

### 5. Structure-Aware Document Ingestion
Unlike traditional sliding-window chunking that breaks legal articles in half, this system uses **Hierarchical Parsing via Docling**. 
- It understands PDF and XML structures to preserve document hierarchy (sections, paragraphs, and articles), resulting in semantically complete context chunks.

### 6. Web-Augmented Question Answering
To handle Out-of-Domain queries or queries requiring the latest real-world info:
- **Exa:** Automatically triggered as a semantic web search fallback if the local vector search yields low confidence scores.
- **Tavily:** Available as a user-triggered manual web search.

## 📊 Evaluation Summary

The performance of the system has been rigorously evaluated. Automated metrics on the primary benchmark run (with Exa fallback enabled) yield:

- **Precision@5:** 0.7524
- **Recall@5:** 0.9405
- **Mean Reciprocal Rank (MRR):** 0.8786
- **BERTScore F1:** 0.9472 (measuring semantic generation quality)
- **Failures/Hallucinations:** 0

*Reranking substantially improved retrieval versus the baseline hybrid search, while web fallback effectively covered out-of-domain queries.*

## 🚀 How to Run It

### Prerequisites
- Docker and Docker Compose installed.
- Required API keys configured.

### Steps

1. **Clone the repository:**
   ```bash
   git clone git@github.com:D3epX/Agentic-Legal-RAG-DZ.git
   cd Agentic-Legal-RAG-DZ
   ```

2. **Configure Environment Variables:**
   A `.env` file is required. You can duplicate the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```
   *Fill in your PostgreSQL credentials, Groq/Gemini API keys, and Exa/Tavily API keys in `.env`.*

3. **Build and Run with Docker Compose:**
   ```bash
   docker compose up --build -d
   ```

4. **Access the Chatbot:**
   - **Chat UI:** `http://localhost:8000`
   - **API Health Check:** `http://localhost:8000/health`

## 📝 Repository Notes
- The User Interface has been customized for this repository to be **Legal-only**.
- Unnecessary platform navigation and NLP/AI modes have been stripped out to provide a unified, focused advisory experience.
