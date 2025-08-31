
# Doc Q&A (RAG demo)

Minimal PDF Q&A using LangChain (JS), OpenAI embeddings, and an in-memory vector store.

## Quickstart
1. Install Node 18+.
2. Create `.env` from `.env.example` and set `OPENAI_API_KEY`.
3. Install deps and run:
   ```bash
   npm i
   npm start
   ```
4. Open http://localhost:3001 and:
   - Upload a PDF
   - Ask questions

## Notes
- Uses `MemoryVectorStore` for simplicity. Swap to FAISS/HNSWLIB for persistence.
- Embeddings: `text-embedding-3-small`. LLM: `gpt-4o-mini` (change in `server.mjs` if needed).
