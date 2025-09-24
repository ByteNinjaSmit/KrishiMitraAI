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
import hashlib
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

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
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

    @staticmethod
    def deduplicate_documents(documents: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []
        for doc in documents:
            hash_val = hashlib.md5(doc.page_content.strip().lower().encode()).hexdigest()
            if hash_val not in seen:
                seen.add(hash_val)
                unique_docs.append(doc)
        return unique_docs


    def _build_or_load_index(self):
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.build_status["device"] = device

        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",   # stronger retrieval model
            model_kwargs={"device": device}
        )


        os.makedirs(self.index_dir, exist_ok=True)

        # Separate FAQ vs Policy
        faq_docs = [d for d in self.documents if d.metadata.get("document_type") == "faq"]
        policy_docs = [d for d in self.documents if d.metadata.get("document_type") == "policy"]

        # Deduplicate
        faq_docs = self.deduplicate_documents(faq_docs)
        policy_docs = self.deduplicate_documents(policy_docs)

        # Load or build FAQ index
        faq_path = os.path.join(self.index_dir, "faq")
        try:
            self.faq_vectorstore = FAISS.load_local(faq_path, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing FAQ index")
        except Exception:
            self.faq_vectorstore = FAISS.from_documents(faq_docs, embeddings) if faq_docs else None
            if self.faq_vectorstore:
                self.faq_vectorstore.save_local(faq_path)

        # Load or build Policy index
        policy_path = os.path.join(self.index_dir, "policy")
        try:
            self.policy_vectorstore = FAISS.load_local(policy_path, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing Policy index")
        except Exception:
            self.policy_vectorstore = FAISS.from_documents(policy_docs, embeddings) if policy_docs else None
            if self.policy_vectorstore:
                self.policy_vectorstore.save_local(policy_path)

        self.build_status["vectorstore_ready"] = True
        self.build_status["last_message"] = "Vector stores (FAQ + Policy) ready"




    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = [t for t in text.replace("\n", " ").split(" ") if len(t) > 2 and t.isascii()]
        return tokens[:200]

    def _build_keyword_index(self):
        """Build BM25 index for fallback retrieval"""
        try:
            self._doc_texts = [doc.page_content for doc in self.documents]
            tokenized_corpus = [self._tokenize(text) for text in self._doc_texts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.keyword_index_ready = True
        except Exception:
            self.keyword_index_ready = False

    def _keyword_fallback(self, query: str, k: int = 5) -> List[Document]:
        if not self.keyword_index_ready:
            return []
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx if scores[i] > 0]

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant documents with FAQ priority, policy fallback,
        deduplication, and Kerala-specific boosting.
        Falls back to keyword search if vectors unavailable.
        """

        faq_hits, policy_hits = [], []
        try:
            if getattr(self, "faq_vectorstore", None):
                faq_hits = self.faq_vectorstore.similarity_search(query, k=k)
            if getattr(self, "policy_vectorstore", None):
                policy_hits = self.policy_vectorstore.similarity_search(query, k=max(2, k // 2))
        except Exception as e:
            print(f"Vector search failed: {e}")

        # If no vector hits, fallback to keyword index
        if not faq_hits and not policy_hits:
            return self._keyword_fallback(query, k)

        # --- Kerala boosting ---
        def boost_score(doc: Document, base: float) -> float:
            text = doc.page_content.lower()
            if "kerala" in text:
                base *= 1.5
            return base

        scored_docs = []
        for i, d in enumerate(faq_hits):
            scored_docs.append((boost_score(d, 1.0 - i*0.01), d))
        for i, d in enumerate(policy_hits):
            scored_docs.append((boost_score(d, 0.7 - i*0.01), d))

        # Deduplicate
        seen, unique_docs = set(), []
        for score, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True):
            key = (doc.metadata.get("file_path"), doc.metadata.get("chunk_id"), doc.page_content[:120])
            if key in seen:
                continue
            seen.add(key)
            unique_docs.append(doc)
            if len(unique_docs) >= k:
                break

        if unique_docs:
            rerank_inputs = [(query, d.page_content) for d in unique_docs]
            scores = self.reranker.predict(rerank_inputs)
            reranked = sorted(zip(scores, unique_docs), key=lambda x: x[0], reverse=True)
            unique_docs = [d for _, d in reranked[:k]]

        return unique_docs

    def _keyword_fallback(self, query: str, k: int = 5) -> List[Document]:
        """Fallback retrieval when vector stores fail"""
        if not self.keyword_index_ready:
            return []
        tokens = self._tokenize(query)
        candidate_ids = set()
        for t in tokens:
            candidate_ids.update(self.keyword_index.get(t, []))
        scores = []
        for i in candidate_ids:
            text = self._doc_texts[i].lower()
            score = sum(text.count(t) for t in tokens)
            if score > 0:
                scores.append((score, self.documents[i]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scores[:k]]


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
        """Prepare context with clear FAQ vs Policy labeling"""
        context_parts = []
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue
            dtype = doc.metadata.get("document_type", "doc")
            if dtype == "faq":
                label = f"FAQ Answer {i+1}"
            elif dtype == "policy":
                label = f"Policy Reference {i+1}"
            else:
                label = f"Document {i+1}"
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
