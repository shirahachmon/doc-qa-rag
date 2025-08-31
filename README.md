# 📘 Doc-QA-RAG

Ask questions about your own PDFs using Retrieval-Augmented Generation (RAG).

## ✨ Features
- Upload any PDF document
- Extract text with [pdfjs-dist](https://www.npmjs.com/package/pdfjs-dist)
- Chunk and embed with HuggingFace embeddings
- Store in memory vector DB
- Ask natural language questions → get answers strictly from your document

## 🛠 Tech Stack
- Node.js + Express
- LangChain (text splitter, vector store)
- HuggingFace Inference API (Embeddings + LLM)
- pdfjs-dist

## 🚀 Getting Started

```bash
git clone https://github.com/your-username/doc-qa-rag.git
cd doc-qa-rag

npm install

# create .env file
echo "HUGGINGFACEHUB_API_KEY=hf_xxxxxxxxxx" > .env

npm start
