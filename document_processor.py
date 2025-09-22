import pandas as pd
import PyPDF2
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    def __init__(self):
        """Initialize document processor with text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF document and return list of Document objects"""
        documents = []
        
        try:
            # Read PDF file
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                full_text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            full_text += f"\n\nPage {page_num + 1}:\n{page_text}"
                    except Exception as e:
                        print(f"Error reading page {page_num + 1}: {str(e)}")
                        continue
                
                # Split text into chunks
                if full_text.strip():
                    text_chunks = self.text_splitter.split_text(full_text)
                    
                    # Create Document objects
                    for i, chunk in enumerate(text_chunks):
                        if chunk.strip():
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": "Kerala Agricultural Development Policy 2015",
                                    "document_type": "policy",
                                    "chunk_id": i,
                                    "file_path": file_path
                                }
                            )
                            documents.append(doc)
            
            print(f"Successfully processed PDF: {len(documents)} chunks created")
            return documents
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
            return []

    def process_csv(self, file_path: str) -> List[Document]:
        """Process CSV document and return list of Document objects.
        Supports flexible column names for question/answer fields.
        """
        documents = []
        
        try:
            # Read CSV file with common encodings
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')

            # Normalize column names
            df.columns = [str(c).strip().lower() for c in df.columns]

            # Candidate columns for questions and answers
            question_cols = [
                'question', 'questions', 'query', 'prompt', 'ques', 'q'
            ]
            answer_cols = [
                'answer', 'answers', 'response', 'reply', 'ans', 'a'
            ]

            def find_first_existing(candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            q_col = find_first_existing(question_cols)
            a_col = find_first_existing(answer_cols)

            if q_col is None or a_col is None:
                print(f"CSV {os.path.basename(file_path)} missing question/answer columns. Columns: {list(df.columns)}")
                return []

            df = df.fillna("")

            # Process each row as a Q&A pair
            for index, row in df.iterrows():
                try:
                    question = str(row.get(q_col, '')).strip()
                    answer = str(row.get(a_col, '')).strip()
                    if not question or not answer:
                        continue
                    # Create content with both question and answer
                    content = f"Question: {question}\nAnswer: {answer}"
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": os.path.basename(file_path) if os.path.basename(file_path) else "Farming FAQ Assistant Dataset",
                            "document_type": "faq",
                            "question": question,
                            "answer": answer,
                            "row_id": index,
                            "file_path": file_path
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error processing row {index} in {file_path}: {str(e)}")
                    continue
            
            print(f"Successfully processed CSV {os.path.basename(file_path)}: {len(documents)} FAQ pairs created")
            return documents
            
        except Exception as e:
            print(f"Error processing CSV {file_path}: {str(e)}")
            return []

    def process_text_file(self, file_path: str) -> List[Document]:
        """Process plain text file and return list of Document objects"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                if content.strip():
                    # Split text into chunks
                    text_chunks = self.text_splitter.split_text(content)
                    
                    # Create Document objects
                    for i, chunk in enumerate(text_chunks):
                        if chunk.strip():
                            doc = Document(
                                page_content=chunk,
                                metadata={
                                    "source": os.path.basename(file_path),
                                    "document_type": "text",
                                    "chunk_id": i,
                                    "file_path": file_path
                                }
                            )
                            documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error processing text file {file_path}: {str(e)}")
            return []

    def get_document_stats(self, documents: List[Document]) -> dict:
        """Get statistics about processed documents"""
        stats = {
            "total_documents": len(documents),
            "document_types": {},
            "sources": {},
            "avg_content_length": 0
        }
        
        total_length = 0
        for doc in documents:
            # Count by document type
            doc_type = doc.metadata.get("document_type", "unknown")
            stats["document_types"][doc_type] = stats["document_types"].get(doc_type, 0) + 1
            
            # Count by source
            source = doc.metadata.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1
            
            # Calculate total length
            total_length += len(doc.page_content)
        
        if len(documents) > 0:
            stats["avg_content_length"] = total_length / len(documents)
        
        return stats
