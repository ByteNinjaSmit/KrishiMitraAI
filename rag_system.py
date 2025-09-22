import os
import json
from typing import List, Dict
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from google import genai
from google.genai import types
from document_processor import DocumentProcessor
from utils import detect_language, clean_text, format_agricultural_response, get_fallback_message
import numpy as np
import faiss
import ollama
import subprocess

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
import threading

class KrishiMitraRAG:
    def __init__(self):
        self.ollama_model = "llama2-uncensored:7b"
        self.use_ollama = self._check_ollama_available()

        """Initialize the RAG system for Krishi Mitra"""
        # Use GEMINI_API_KEY from environment (your existing Google API key will work)
        # api_key = os.getenv("GEMINI_API_KEY", "")
        api_key = "AIzaSyDx4B06Bpq1Vws_TxpVxD99LIoNckwCy_g"
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required. Please set it in your environment.")
        self.gemini_client = genai.Client(api_key=api_key)
        self.vectorstore = None
        self.document_processor = DocumentProcessor()
        self.index_dir = os.path.join("attached_assets", "index")
        self.build_status = {
            "vectorstore_ready": False,
            "last_message": "Not started",
            "device": None,
        }
        # Lexical index for fast keyword retrieval over the full corpus
        self.keyword_index = {}
        self.keyword_index_ready = False
        self._doc_texts: List[str] = []
        self.system_prompt = """
You are **Krishi Mitra**, an AI-based digital agriculture advisor for farmers in Kerala. 
Your role is to provide accurate, concise, and practical answers to farmers' questions.

### Guidelines:
- Always use the **retrieved documents** (FAQs, Kerala Agri-Dev Policy) as your main knowledge source.
- If the retrieved text mentions specific government schemes, policies, fertilizers, or pest control methods, include them clearly in your answer.
- Prioritize **Kerala-specific policies, subsidies, and practices**.
- Answer in **simple farmer-friendly language**. Avoid jargon.
- If the question is in Malayalam, try to answer in Malayalam. If in English, reply in English.
- If information is not available in the documents, politely say:
  "I don't have official details about that. Please consult your local Krishi Bhavan officer."
- Keep answers short (3–6 sentences). Use bullet points if multiple steps are needed.

### Example Style:
👨‍🌾 Farmer: Which pesticide for leaf spot in banana?  
🤖 Advisor: For banana leaf spot, spray Mancozeb (0.25%) or Carbendazim (0.1%).  
- Keep the field well-drained.  
- Avoid water stagnation.  
(If symptoms continue, consult your Krishi Bhavan.)
"""

    def _check_ollama_available(self) -> bool:
        """Check if Ollama and the specified model are available locally"""
        try:
            # Check if ollama is installed
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False

            # Check if model exists locally
            models = result.stdout.lower()
            return self.ollama_model.lower() in models
        except Exception:
            return False

    def initialize_system(self):
        """Initialize the RAG system by processing documents and creating document store"""
        try:
            # Process documents
            self.documents = self._load_and_process_documents()
            
            # Build or load vector store in a background thread to avoid blocking UI
            if len(self.documents) > 0:
                threading.Thread(target=self._build_or_load_index, daemon=True).start()
                threading.Thread(target=self._build_keyword_index, daemon=True).start()
            else:
                self.vectorstore = None
            
            print("RAG system initialized successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing RAG system: {str(e)}")
            raise e

    def _load_and_process_documents(self) -> List[Document]:
        """Load and process all agricultural documents"""
        all_documents = []
        
        # Process PDF document
        try:
            pdf_docs = self.document_processor.process_pdf("attached_assets/Agri-Dev-Policy_compressed-1-1_1758519620532.pdf")
            all_documents.extend(pdf_docs)
            print(f"Processed PDF: {len(pdf_docs)} documents")
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
        
        # Process CSV document
        try:
            csv_docs = self.document_processor.process_csv("attached_assets/Farming_FAQ_Assistant_Dataset_1758519620533.csv")
            all_documents.extend(csv_docs)
            print(f"Processed CSV: {len(csv_docs)} documents")
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")

        # Process additional questions dataset if present
        if os.path.exists("attached_assets/questionsv4.csv"):
            try:
                extra_docs = self.document_processor.process_csv("attached_assets/questionsv4.csv")
                all_documents.extend(extra_docs)
                print(f"Processed questionsv4 CSV: {len(extra_docs)} documents")
            except Exception as e:
                print(f"Error processing questionsv4.csv: {str(e)}")
        
        return all_documents



    def _build_or_load_index(self):
        """Build or load FAISS index and move it to GPU if available."""
        try:
            # Detect device (GPU if available)
            device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            self.build_status["device"] = device
            self.build_status["last_message"] = f"Preparing embeddings on {device.upper()}"

            # Load embeddings on GPU
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": device}
            )

            os.makedirs(self.index_dir, exist_ok=True)

            try:
                self.build_status["last_message"] = "Loading existing FAISS index"
                self.vectorstore = FAISS.load_local(
                    self.index_dir, embeddings, allow_dangerous_deserialization=True
                )
            except Exception:
                self.build_status["last_message"] = "Building FAISS index (first run)"
                self.vectorstore = FAISS.from_documents(self.documents, embeddings)

                try:
                    self.vectorstore.save_local(self.index_dir)
                except Exception:
                    pass

            # 🚀 Move FAISS index to GPU if available
            if torch.cuda.is_available():
                import faiss
                res = faiss.StandardGpuResources()
                cpu_index = self.vectorstore.index
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                self.vectorstore.index = gpu_index
                self.build_status["last_message"] = "Vector store ready on GPU"
            else:
                self.build_status["last_message"] = "Vector store ready on CPU"

            self.build_status["vectorstore_ready"] = True

        except Exception as e:
            self.build_status["last_message"] = f"Index build error: {str(e)}"


    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = [t for t in text.replace("\n", " ").split(" ") if len(t) > 2 and t.isascii()]
        return tokens[:200]

    def _build_keyword_index(self):
        """Build a lightweight inverted index across all docs for full-corpus fallback."""
        try:
            self.keyword_index = {}
            self._doc_texts = []
            for idx, doc in enumerate(self.documents):
                content = doc.page_content
                self._doc_texts.append(content)
                for tok in set(self._tokenize(content)):
                    postings = self.keyword_index.get(tok)
                    if postings is None:
                        self.keyword_index[tok] = [idx]
                    else:
                        if len(postings) < 800:  # cap postings to keep memory bounded
                            postings.append(idx)
            self.keyword_index_ready = True
        except Exception:
            self.keyword_index_ready = False

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for the given query using vector similarity (FAISS),
        falling back to simple keyword scoring if the vector store is unavailable."""
        if not hasattr(self, 'documents'):
            raise ValueError("Document store not initialized. Call initialize_system() first.")
        
        # Start with vector results if available
        vector_results: List[Document] = []
        if self.vectorstore is not None and self.build_status.get("vectorstore_ready", False):
            try:
                vector_results = self.vectorstore.similarity_search(query, k=max(k, 6))
            except Exception as e:
                # Fall back to keyword method on error
                print(f"Vector search failed, falling back to keyword search: {str(e)}")
        
        # Keyword scoring via inverted index if available, else fallback to simple scan sample
        keyword_results: List[Document] = []
        if self.keyword_index_ready and len(self._doc_texts) == len(self.documents):
            tokens = self._tokenize(query)
            candidate_ids = set()
            for t in tokens:
                plist = self.keyword_index.get(t)
                if plist:
                    candidate_ids.update(plist)
            # Score candidates by term frequency
            scores = []
            for i in candidate_ids:
                text = self._doc_texts[i].lower()
                score = sum(text.count(t) for t in tokens)
                if score > 0:
                    scores.append((score, i))
            scores.sort(key=lambda x: x[0], reverse=True)
            for _, i in scores[: max(k, 8)]:
                keyword_results.append(self.documents[i])
        else:
            query_lower = query.lower()
            scored_docs = []
            corpus = self.documents if len(self.documents) <= 10000 else self.documents[:10000]
            for doc in corpus:
                content_lower = doc.page_content.lower()
                score = 0
                for word in query_lower.split():
                    if len(word) > 2:
                        score += content_lower.count(word)
                if score > 0:
                    scored_docs.append((score, doc))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            keyword_results = [doc for _, doc in scored_docs[:k]]

        # Merge unique docs preserving order: vector first then keyword
        seen = set()
        merged: List[Document] = []
        for d in list(vector_results) + list(keyword_results):
            key = (d.metadata.get("file_path"), d.metadata.get("chunk_id"), d.page_content[:80])
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
            if len(merged) >= max(k, 8):
                break
        return merged

    def get_response(self, query: str) -> str:
        """Get response from Krishi Mitra using RAG with either Ollama (if available) or Gemini"""
        try:
            relevant_docs = self.retrieve_relevant_docs(query, k=8)
            context = self._prepare_context(relevant_docs)
            dataset_facts = self._extract_structured_facts(relevant_docs)
            detected_lang = detect_language(query)

            prompt = self._prepare_prompt(query, context, detected_lang, dataset_facts)
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

            # ---- Prefer Ollama if available ----
            if self.use_ollama:
                try:
                    response = ollama.chat(
                        model=self.ollama_model,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    content = response["message"]["content"].strip()
                except Exception as e:
                    print(f"Ollama failed, falling back to Gemini: {e}")
                    self.use_ollama = False
                    return self.get_response(query)  # retry with Gemini

            # ---- Otherwise use Gemini ----
            else:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                content = response.text if response and getattr(response, 'text', None) else None

            # ---- Post-processing ----
            if not content:
                if dataset_facts:
                    return self._compose_answer_from_facts(query, detected_lang, dataset_facts)
                return get_fallback_message(detected_lang)

            formatted = format_agricultural_response(clean_text(content))
            if len(formatted.strip()) < 10 and dataset_facts:
                return self._compose_answer_from_facts(query, detected_lang, dataset_facts)

            return formatted

        except Exception as e:
            lang = detect_language(query)
            try:
                relevant_docs = self.retrieve_relevant_docs(query, k=8)
                facts = self._extract_structured_facts(relevant_docs)
                if facts:
                    return self._compose_answer_from_facts(query, lang, facts)
            except Exception:
                pass
            return get_fallback_message(lang)


    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if content:
                source = doc.metadata.get("source", "Unknown source")
                label = f"Document {i+1} ({source})"
                context_parts.append(f"{label}:\n{content}")
        
        return "\n\n".join(context_parts)

    def _prepare_prompt(self, query: str, context: str, language: str, dataset_facts: List[str] = None) -> str:
        """Prepare the final prompt for the language model"""
        language_instruction = ""
        if language == "malayalam":
            language_instruction = "Please respond in Malayalam as the farmer asked in Malayalam."
        elif language == "english":
            language_instruction = "Please respond in English as the farmer asked in English."
        facts_block = ""
        if dataset_facts:
            bullet_facts = "\n".join([f"- {f}" for f in dataset_facts[:6]])
            facts_block = f"\n\nVerified facts from FAQ dataset (use directly if relevant):\n{bullet_facts}\n"
        
        prompt = f"""
Context from Agricultural Documents (Kerala-specific):
{context}

Farmer's Question: {query}

{language_instruction}
{facts_block}

Rules:
- Answer ONLY using the context and facts provided above. If multiple sources agree, combine briefly.
- Prefer Kerala context and seasonality. Keep 3–6 sentences, with bullet points for steps.
- If the information is not in the context/facts, say you don't have official details.

Provide the answer now.
"""
        return prompt

    def _extract_structured_facts(self, documents: List[Document]) -> List[str]:
        """Extract concise facts from FAQ-type documents within retrieved set.
        Returns short, declarative statements suitable for direct inclusion."""
        facts: List[str] = []
        for doc in documents:
            if doc.metadata.get("document_type") == "faq":
                q = str(doc.metadata.get("question", "")).strip()
                a = str(doc.metadata.get("answer", "")).strip()
                if a:
                    # Prefer answer as a single-line fact; trim overly long strings
                    line = a.split("\n")[0].strip()
                    if len(line) > 260:
                        line = line[:260] + "..."
                    facts.append(line)
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for f in facts:
            if f in seen:
                continue
            seen.add(f)
            uniq.append(f)
        return uniq[:10]

    def _compose_answer_from_facts(self, query: str, language: str, facts: List[str]) -> str:
        """Compose a concise answer directly from dataset facts when model output is unavailable."""
        if not facts:
            return get_fallback_message(language)
        # Prefer the first 1-3 facts
        top = facts[:3]
        if language == "malayalam":
            lines = "\n".join([f"- {t}" for t in top])
            return f"ഇതാ രേഖാസര്‍ത്തമായ വിവരം:\n{lines}"
        else:
            lines = "\n".join([f"- {t}" for t in top])
            return f"Based on our Kerala agriculture FAQ, here is what applies:\n{lines}"

    def add_documents(self, documents: List[Document]):
        """Add new documents to the existing document store"""
        if not hasattr(self, 'documents'):
            raise ValueError("Document store not initialized. Call initialize_system() first.")
        
        self.documents.extend(documents)

    def _format_citations(self, documents: List[Document]) -> str:
        """Create a simple citation list from document metadata."""
        lines = []
        for i, doc in enumerate(documents):
            source = doc.metadata.get("source", "Unknown source")
            dtype = doc.metadata.get("document_type", "doc")
            extra = ""
            if dtype == "faq":
                q = doc.metadata.get("question")
                if q:
                    extra = f" – Q: {q[:80]}..." if len(q) > 80 else f" – Q: {q}"
            lines.append(f"{i+1}. {source}{extra}")
        return "\n".join(lines)

    def analyze_crop_image(self, image_path: str) -> str:
        """Analyze crop image for disease identification and treatment recommendations"""
        try:
            # Read image file
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Prepare specialized prompt for crop disease analysis
            crop_analysis_prompt = """
You are Krishi Mitra, an expert agricultural advisor specializing in crop disease identification for Kerala farmers.

Analyze this crop image and provide:
1. **Plant/Crop Identification**: What crop or plant is this?
2. **Health Assessment**: Is the plant healthy or showing signs of disease/pest damage?
3. **Disease/Problem Identification**: If unhealthy, identify the specific disease, pest, or nutrient deficiency
4. **Treatment Recommendations**: Provide specific treatment methods, including:
   - Organic/natural treatments preferred
   - Chemical treatments if necessary (with proper safety instructions)
   - Preventive measures
5. **Timeline**: When to expect improvement and follow-up actions

Keep your response practical and farmer-friendly. If you cannot clearly identify the issue, recommend consulting the local Krishi Bhavan officer.

Focus on common Kerala crops like coconut, rice, pepper, banana, rubber, tea, coffee, cardamom, etc.
"""
            
            # Use Gemini's vision capability
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-pro",  # Pro model for better vision analysis
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg",
                    ),
                    crop_analysis_prompt
                ]
            )
            
            if response.text:
                return response.text
            else:
                return "I couldn't analyze the image properly. Please ensure the image is clear and shows the crop clearly, then try again. If the problem persists, consult your local Krishi Bhavan officer."
                
        except Exception as e:
            return f"Error analyzing crop image: {str(e)}. Please try uploading a different image or consult your local Krishi Bhavan officer."
