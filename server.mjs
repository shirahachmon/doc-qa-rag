import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import multer from "multer";
import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

// LangChain (split + vector store + HF embeddings)
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";

// Hugging Face Inference SDK (chat)
import { HfInference } from "@huggingface/inference";

dotenv.config();

const app = express();
const upload = multer();
app.use(cors());
app.use(express.json());

// Static UI
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(path.join(__dirname, "public")));

// --- pdfjs-dist (CommonJS) ---
const require = createRequire(import.meta.url);
const pdfjsLib = require("pdfjs-dist/legacy/build/pdf.js");
const pdfjsWorker = require("pdfjs-dist/legacy/build/pdf.worker.js");

// קישור worker (לא חובה ב-Node, אך טוב להגדיר)
if (pdfjsLib.GlobalWorkerOptions) {
  pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;
}

// נתיב לפונטים הסטנדרטיים (משתיק אזהרות LiberationSans)
const standardFontDataUrl = path.join(
  path.dirname(require.resolve("pdfjs-dist/package.json")),
  "standard_fonts/"
);

// -------- PDF → text --------
async function extractTextFromPDF(buffer) {
  const loadingTask = pdfjsLib.getDocument({
    data: new Uint8Array(buffer), // חשוב: לא Buffer
    disableFontFace: true,        // אין צורך ברנדרינג פונט/קנבס
    standardFontDataUrl           // מספק קבצי פונטים סטנדרטיים
  });
  const pdf = await loadingTask.promise;
  let fullText = "";
  for (let p = 1; p <= pdf.numPages; p++) {
    const page = await pdf.getPage(p);
    const content = await page.getTextContent();
    const strings = content.items.map(it => it.str);
    fullText += strings.join(" ") + "\n";
  }
  return fullText;
}

// ---------- Hugging Face ----------
if (!process.env.HUGGINGFACEHUB_API_KEY) {
  console.warn("⚠️ Missing HUGGINGFACEHUB_API_KEY in .env — requests will fail.");
}
const hf = new HfInference(process.env.HUGGINGFACEHUB_API_KEY);

// Vector store בזיכרון (לנוחות)
let vectorStore = null;
let chunkCount = 0;

/**
 * אינדוקס: העלאת PDF → חילוץ טקסט → חלוקה לצ'אנקים → אמבדינגס HF → VectorStore בזיכרון
 */
app.post("/api/index", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No file uploaded." });

    const text = (await extractTextFromPDF(req.file.buffer)) || "";
    if (!text.trim()) return res.status(400).json({ error: "No textual content found in PDF." });

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 2000,
      chunkOverlap: 300
    });
    const chunks = await splitter.splitText(text);
    const metadatas = chunks.map((_, i) => ({ chunk: i }));
    chunkCount = chunks.length;

    const embeddings = new HuggingFaceInferenceEmbeddings({
      apiKey: process.env.HUGGINGFACEHUB_API_KEY,
      model: "sentence-transformers/all-MiniLM-L6-v2"
    });

    vectorStore = await MemoryVectorStore.fromTexts(chunks, metadatas, embeddings);

    res.json({ ok: true, chunks: chunkCount, provider: "HuggingFace" });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || "Indexing failed." });
  }
});

/**
 * שאילתא: חיפוש דמיון → בניית CONTEXT → צ'אט HF עם הנחיית מערכת קשוחה ל-RAG
 */
app.post("/api/ask", async (req, res) => {
  try {
    if (!vectorStore) return res.status(400).json({ error: "No index yet. Upload a PDF first." });

    const { question } = req.body || {};
    if (!question?.trim()) return res.status(400).json({ error: "Missing 'question'." });

    const docs = await vectorStore.similaritySearch(question, 4);
    const context = docs.map(d => `Chunk ${d.metadata?.chunk}:\n${d.pageContent}`).join("\n\n");

    const system = `You are a RAG question answering bot.
You MUST answer ONLY using the text provided in CONTEXT.
Never invent, never guess. If the answer is not literally in CONTEXT, reply exactly: "I don't know".
Answer concisely.`;

    const user = `QUESTION: ${question}\n\nCONTEXT:\n${context}\n`;

    const resp = await hf.chatCompletion({
      model: "HuggingFaceH4/zephyr-7b-beta", // ניתן להחליף למודל אחר
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ],
      max_tokens: 512,
      temperature: 0.2
    });

    const answer = resp.choices?.[0]?.message?.content ?? "(no content)";
    res.json({
      answer,
      sources: docs.map(d => ({ chunk: d.metadata?.chunk })),
      chunks: chunkCount,
      llmProvider: "HuggingFace Inference"
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message || "Ask failed." });
  }
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`✅ Doc-QA server running on http://localhost:${PORT}`));
