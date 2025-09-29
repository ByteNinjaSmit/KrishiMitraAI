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
    MAX_CHARS = 700   # Limit context chunk size for LLM
    SYNONYMS = {
        "rice": ["paddy"],
        "sow": ["plant"],
        "fertilizer": ["manure", "nutrient", "fertiliser"],  # spelling variant
        "chili": ["chilli", "capsicum"],
        "coconut": ["copra"],
    }

    def __init__(self):
        # self.ollama_model = "gemma3:12b"
        self.ollama_model ="llama3.2:3b"
        self.use_ollama = self._check_ollama_available()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # api_key = os.getenv("GEMINI_API_KEY", "YOUR_FALLBACK_KEY_HERE")
        api_key = "AIzaSyDx4B06Bpq1Vws_TxpVxD99LIoNckwCy_g"
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        self.gemini_client = genai.Client(api_key=api_key)

        self.vectorstore = None
        self.document_processor = DocumentProcessor()
        self.index_dir = os.path.join("attached_assets", "index")
        self.build_status = {
            "vectorstore_ready": False,
            "last_message": "Not started",
            "device": None,
        }

        self._doc_texts: List[str] = []
        self.keyword_index_ready = False

        self.system_prompt = """
You are **Krishi Mitra**, an AI-based agriculture advisor for farmers in Kerala.

### Guidelines:
- Use the **retrieved FAQ & policy docs** as main knowledge.
- Highlight govt schemes, subsidies, fertilizers, pest control when mentioned.
- Prioritize **Kerala-specific context**.
- Answer in simple farmer-friendly language.
- Reply in Malayalam if farmer asks in Malayalam; otherwise English.
- If answer not found, say: "I don't have official details. Please consult your local Krishi Bhavan officer."
- Keep answers short (3–6 sentences). Use bullet points if multiple steps.

### Example:
👨‍🌾: Which pesticide for leaf spot in banana?  
🤖: For banana leaf spot:  
- Spray Mancozeb (0.25%) or Carbendazim (0.1%)  
- Keep field well-drained  
- Avoid water stagnation  
(If symptoms continue, consult Krishi Bhavan)
"""

    # ----------------- System Setup -----------------
    def _check_ollama_available(self) -> bool:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return False
            return self.ollama_model.lower() in result.stdout.lower()
        except Exception:
            return False

    def initialize_system(self):
        try:
            self.documents = self._load_and_process_documents()
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
        all_documents = []
        try:
            pdf_docs = self.document_processor.process_pdf("attached_assets/Agri-Dev-Policy_compressed-1-1_1758519620532.pdf")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")

        try:
            csv_docs = self.document_processor.process_csv("attached_assets/Farming_FAQ_Assistant_Dataset_1758519620533.csv")
            all_documents.extend(csv_docs)
        except Exception as e:
            print(f"Error processing CSV: {str(e)}")

        if os.path.exists("attached_assets/questionsv4.csv"):
            try:
                extra_docs = self.document_processor.process_csv("attached_assets/questionsv4.csv")
                all_documents.extend(extra_docs)
            except Exception as e:
                print(f"Error processing questionsv4.csv: {str(e)}")

        return all_documents

    @staticmethod
    def deduplicate_documents(documents: List[Document]) -> List[Document]:
        seen = set()
        unique_docs = []
        for doc in documents:
            h = hashlib.md5(doc.page_content.strip().lower().encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_docs.append(doc)
        return unique_docs

    def _build_or_load_index(self):
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.build_status["device"] = device

        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": device}
        )
        os.makedirs(self.index_dir, exist_ok=True)

        faq_docs = [d for d in self.documents if d.metadata.get("document_type") == "faq"]
        policy_docs = [d for d in self.documents if d.metadata.get("document_type") == "policy"]

        faq_docs = self.deduplicate_documents(faq_docs)
        policy_docs = self.deduplicate_documents(policy_docs)

        try:
            self.faq_vectorstore = FAISS.load_local(os.path.join(self.index_dir, "faq"), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            self.faq_vectorstore = FAISS.from_documents(faq_docs, embeddings) if faq_docs else None
            if self.faq_vectorstore:
                self.faq_vectorstore.save_local(os.path.join(self.index_dir, "faq"))

        try:
            self.policy_vectorstore = FAISS.load_local(os.path.join(self.index_dir, "policy"), embeddings, allow_dangerous_deserialization=True)
        except Exception:
            self.policy_vectorstore = FAISS.from_documents(policy_docs, embeddings) if policy_docs else None
            if self.policy_vectorstore:
                self.policy_vectorstore.save_local(os.path.join(self.index_dir, "policy"))

        self.build_status["vectorstore_ready"] = True
        self.build_status["last_message"] = "Vector stores ready"

    # ----------------- Retrieval -----------------
    def _tokenize(self, text: str) -> List[str]:
        return [t for t in text.lower().split() if len(t) > 2 and t.isascii()]

    def _build_keyword_index(self):
        try:
            self._doc_texts = [doc.page_content for doc in self.documents]
            tokenized = [self._tokenize(t) for t in self._doc_texts]
            self.bm25 = BM25Okapi(tokenized)
            self.keyword_index_ready = True
        except Exception as e:
            print("BM25 build failed:", e)
            self.keyword_index_ready = False

    def expand_query(self, query: str) -> List[str]:
        expansions = [query]
        for word, alts in self.SYNONYMS.items():
            if word in query.lower():
                for alt in alts:
                    expansions.append(query.lower().replace(word, alt))
        return list(set(expansions))

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        expansions = self.expand_query(query)

        hits = []
        for q in expansions:
            if getattr(self, "faq_vectorstore", None):
                hits.extend(self.faq_vectorstore.similarity_search(q, k=3))
            if getattr(self, "policy_vectorstore", None):
                hits.extend(self.policy_vectorstore.similarity_search(q, k=2))
            if self.keyword_index_ready:
                hits.extend(self._keyword_fallback(q, k=2))

        if not hits:
            return []

        def boost(doc: Document, base: float) -> float:
            text = doc.page_content.lower()
            if "kerala" in text:
                base *= 1.5
            for token in query.lower().split():
                if token in text:
                    base *= 1.2
            return base

        scored = []
        for i, d in enumerate(hits):
            scored.append((boost(d, 1.0 - i * 0.01), d))

        seen, unique = set(), []
        for score, doc in sorted(scored, key=lambda x: x[0], reverse=True):
            key = (doc.metadata.get("file_path"), doc.metadata.get("chunk_id"), doc.page_content[:120])
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        if not unique:
            return []

        rerank_inputs = [(query, d.page_content) for d in unique]
        scores = self.reranker.predict(rerank_inputs)
        reranked = sorted(zip(scores, unique), key=lambda x: x[0], reverse=True)
        return [d for _, d in reranked[:k]]

    def _keyword_fallback(self, query: str, k: int = 5) -> List[Document]:
        if not self.keyword_index_ready:
            return []
        q_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(q_tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx if scores[i] > 0]

    # ----------------- Response -----------------
    # ----------------- Response -----------------
    def get_response(self, query: str) -> str:
        try:
            # Retrieve documents + structured facts
            relevant_docs = self.retrieve_relevant_docs(query, k=8)
            dataset_facts = self._extract_structured_facts(relevant_docs)
            lang = detect_language(query)

            # Always prepare context
            context = self._prepare_context(relevant_docs)
            prompt = self._prepare_prompt(query, context, lang, dataset_facts)
            full_prompt = f"{self.system_prompt}\n\n{prompt}"

            # --- Prefer Ollama if available ---
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
                    print("Ollama failed:", e)
                    self.use_ollama = False
                    return self.get_response(query)  # retry with Gemini
            else:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt
                )
                content = response.text if response and getattr(response, 'text', None) else None

            # --- Post-processing ---
            if not content:
                # If LLM fails, fallback to FAQ facts if any
                if dataset_facts:
                    return self._compose_answer_from_facts(query, lang, dataset_facts)
                return get_fallback_message(lang)

            formatted = format_agricultural_response(clean_text(content))

            # If LLM answer is too short/weak → fallback to FAQ facts
            if len(formatted.strip()) < 10 and dataset_facts:
                return self._compose_answer_from_facts(query, lang, dataset_facts)

            return self._post_process_answer(formatted, dataset_facts, lang)

        except Exception as e:
            lang = detect_language(query)
            try:
                # Last-resort: FAQ facts
                facts = self._extract_structured_facts(self.retrieve_relevant_docs(query, k=8))
                if facts:
                    return self._compose_answer_from_facts(query, lang, facts)
            except Exception:
                pass
            return get_fallback_message(lang)


    def _prepare_context(self, documents: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents):
            content = doc.page_content.strip()
            if not content:
                continue
            if len(content) > self.MAX_CHARS:
                content = content[:self.MAX_CHARS] + "..."
            dtype = doc.metadata.get("document_type", "doc")
            label = "FAQ" if dtype == "faq" else "Policy" if dtype == "policy" else "Doc"
            parts.append(f"{label} {i+1}:\n{content}")
        return "\n\n".join(parts)

    def _prepare_prompt(self, query: str, context: str, lang: str, dataset_facts: List[str] = None) -> str:
        lang_instruction = "Please respond in Malayalam." if lang == "malayalam" else "Please respond in English."
        facts_block = ""
        if dataset_facts:
            facts_block = "\n\nVerified facts from FAQ:\n" + "\n".join([f"- {f}" for f in dataset_facts[:5]])
        return f"""
Context:
{context}

Farmer's Question: {query}

{lang_instruction}
{facts_block}

Rules:
- Use only the context and facts.
- Answer in ≤ 6 sentences, bullet points if possible.
- Format:
  * Recommended practice:
  * Fertilizer/Pesticide (if applicable):
  * Extra notes:
- If unsure, say: "I don't have official details. Please consult your local Krishi Bhavan officer."
"""

    def _extract_structured_facts(self, documents: List[Document]) -> List[str]:
        facts = []
        for doc in documents:
            if doc.metadata.get("document_type") == "faq":
                ans = str(doc.metadata.get("answer", "")).strip()
                if ans:
                    line = ans.split("\n")[0].strip()
                    facts.append(line[:260] + "..." if len(line) > 260 else line)
        seen, uniq = set(), []
        for f in facts:
            if f not in seen:
                seen.add(f)
                uniq.append(f)
        return uniq[:10]

    def _compose_answer_from_facts(self, query: str, lang: str, facts: List[str]) -> str:
        if not facts:
            return get_fallback_message(lang)
        lines = "\n".join([f"- {f}" for f in facts[:3]])
        return f"ഇതാ രേഖാസര്‍ത്തമായ വിവരം:\n{lines}" if lang == "malayalam" else f"Based on our Kerala agriculture FAQ:\n{lines}"

    def _post_process_answer(self, answer: str, facts: List[str], lang: str) -> str:
        sentences = answer.split(". ")
        if len(sentences) > 6:
            answer = ". ".join(sentences[:6])
        if len(answer.strip()) < 10 and facts:
            return self._compose_answer_from_facts("", lang, facts)
        return answer
    
        # ----------------- Crop Disease Detection -----------------
    def analyze_crop_image(self, image_path: str) -> str:
        """
        Analyze a crop image using Gemini multimodal model.
        Returns disease identification and treatment recommendations.
        """
        try:
            # Open image in binary
            with open(image_path, "rb") as img_file:
                image_bytes = img_file.read()

            # Send image + instruction to Gemini
            response = self.gemini_client.models.generate_content(
                model="gemini-1.5-flash",   # use a multimodal model
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"mime_type": "image/jpeg", "data": image_bytes},
                            {
                                "text": """You are Krishi Mitra, an agriculture expert for Kerala farmers.
Identify any visible crop disease, pest, or deficiency in this image.
Then give:
- Name of disease/pest (if identifiable)
- Symptoms observed
- Treatment steps (fertilizer, pesticide, or cultural practices)
- Precautionary tips

Keep response short and farmer-friendly.
If unsure, say: "Please consult your local Krishi Bhavan for expert advice." """
                            }
                        ]
                    }
                ]
            )

            if hasattr(response, "text") and response.text.strip():
                return clean_text(response.text.strip())
            else:
                return "Sorry, I could not analyze this image. Please try again with a clearer photo."

        except Exception as e:
            return f"Error analyzing crop image: {str(e)}"

