import os
import json
import uuid
import pandas as pd
import chromadb
from typing import List, Dict, Optional, Any
from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class KnowledgeBase:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.principles_file = os.path.join(storage_dir, "principles.json")
        self.cases_file = os.path.join(storage_dir, "cases.json")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=os.path.join(storage_dir, "chroma_db"))
        # Use default embedding function for simplicity in MVP
        self.principle_collection = self.chroma_client.get_or_create_collection(name="principles")
        self.case_collection = self.chroma_client.get_or_create_collection(name="cases")

        # Initialize JSON files if they don't exist
        if not os.path.exists(self.principles_file):
            with open(self.principles_file, 'w') as f:
                json.dump([], f)
        if not os.path.exists(self.cases_file):
            with open(self.cases_file, 'w') as f:
                json.dump([], f)

    def _load_json(self, filepath: str) -> List[Dict]:
        with open(filepath, 'r') as f:
            return json.load(f)

    def _save_json(self, filepath: str, data: List[Dict]):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def save_principle(self, principle: Dict):
        # 1. Save to JSON
        principles = self._load_json(self.principles_file)
        principles.append(principle)
        self._save_json(self.principles_file, principles)

        # 2. Save to ChromaDB
        # Construct a text representation for embedding
        text_to_embed = f"{principle['summary']}\n{principle['description']}"
        self.principle_collection.add(
            documents=[text_to_embed],
            metadatas=[{"source": principle.get("source", "Unknown"), "type": "principle", "summary": principle["summary"]}],
            ids=[principle["id"]]
        )

    def save_case(self, case: Dict):
        # 1. Save to JSON
        cases = self._load_json(self.cases_file)
        cases.append(case)
        self._save_json(self.cases_file, cases)

        # 2. Save to ChromaDB
        text_to_embed = f"{case['description']}"
        self.case_collection.add(
            documents=[text_to_embed],
            metadatas=[{"source": case.get("source", "Unknown"), "type": case.get("type", "general")}],
            ids=[case["id"]]
        )

    def get_all_principles(self) -> List[Dict]:
        return self._load_json(self.principles_file)

    def get_all_cases(self) -> List[Dict]:
        return self._load_json(self.cases_file)

    def update_principle(self, updated_principle: Dict):
        principles = self._load_json(self.principles_file)
        for i, p in enumerate(principles):
            if p["id"] == updated_principle["id"]:
                principles[i] = updated_principle
                break
        self._save_json(self.principles_file, principles)
        
        # Update ChromaDB (Delete then Add)
        try:
            self.principle_collection.delete(ids=[updated_principle["id"]])
            text_to_embed = f"{updated_principle['summary']}\n{updated_principle['description']}"
            self.principle_collection.add(
                documents=[text_to_embed],
                metadatas=[{"source": updated_principle.get("source", "Unknown"), "type": "principle"}],
                ids=[updated_principle["id"]]
            )
        except:
            pass # Handle case where ID might not exist in Chroma

    def delete_principle(self, principle_id: str):
        principles = self._load_json(self.principles_file)
        principles = [p for p in principles if p["id"] != principle_id]
        self._save_json(self.principles_file, principles)
        
        try:
            self.principle_collection.delete(ids=[principle_id])
        except:
            pass

    def update_case(self, updated_case: Dict):
        cases = self._load_json(self.cases_file)
        for i, c in enumerate(cases):
            if c["id"] == updated_case["id"]:
                cases[i] = updated_case
                break
        self._save_json(self.cases_file, cases)
        
        try:
            self.case_collection.delete(ids=[updated_case["id"]])
            text_to_embed = f"{updated_case['description']}"
            self.case_collection.add(
                documents=[text_to_embed],
                metadatas=[{"source": updated_case.get("source", "Unknown"), "type": updated_case.get("type", "general")}],
                ids=[updated_case["id"]]
            )
        except:
            pass

    def delete_case(self, case_id: str):
        cases = self._load_json(self.cases_file)
        cases = [c for c in cases if c["id"] != case_id]
        self._save_json(self.cases_file, cases)
        
        try:
            self.case_collection.delete(ids=[case_id])
        except:
            pass

class IngestionEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "your_api_key_here" in api_key or "*****" in api_key:
            raise ValueError("Invalid OpenAI API Key. Please check your .env file.")
        self.client = OpenAI(api_key=api_key)

    def extract_text_from_pdf(self, file) -> str:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def extract_text_from_epub(self, file_path) -> str:
        book = epub.read_epub(file_path)
        text = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n"
        return text

class IngestionEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "your_api_key_here" in api_key or "*****" in api_key:
            raise ValueError("Invalid OpenAI API Key. Please check your .env file.")
        self.client = OpenAI(api_key=api_key)

    def extract_text_from_pdf(self, file) -> str:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def extract_text_from_epub(self, file_path) -> str:
        book = epub.read_epub(file_path)
        text = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text += soup.get_text() + "\n"
        return text

    def extract_principles_and_cases(self, text: str, source_name: str) -> Dict[str, int]:
        """
        Extracts principles and cases from text using LLM and saves them.
        Returns a summary of counts.
        """
        # Chunking text to avoid token limits (simplified for MVP: just take first 15k chars or chunk properly)
        # For MVP, let's assume we process a reasonable chunk or the user uploads small files.
        # Real implementation needs robust chunking.
        
        chunk_size = 10000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        total_principles = 0
        total_cases = 0

        system_prompt = """
        You are an expert knowledge engineer. Your goal is to extract "Principles" and "Cases" from the provided text.
        
        CRITICAL INSTRUCTIONS:
        1. **Hierarchy**: Identify the "Parent Context" (e.g., Chapter Title or Section Header) that these principles belong to.
        2. **Translation**: Provide a Chinese translation for the Summary and Description.
        3. **Deduplication**: Be precise. Do not extract duplicate principles if they are just restated.
        
        Output JSON format:
        {
            "principles": [
                {
                    "summary": "Short summary (English)",
                    "summary_cn": "Short summary (Chinese)",
                    "description": "Detailed explanation (English)",
                    "description_cn": "Detailed explanation (Chinese)",
                    "parent_context": "Chapter Title or Section Header"
                }
            ],
            "cases": [
                {
                    "description": "Story or example (English)",
                    "description_cn": "Story or example (Chinese)",
                    "related_principle_summary": "Summary of principle this case illustrates",
                    "type": "Book Example"
                }
            ]
        }
        """

        for i, chunk in enumerate(chunks):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o", # Or gpt-3.5-turbo
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Extract from this text:\n\n{chunk}"}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Save Principles with Deduplication
                for p in data.get("principles", []):
                    # Deduplication Check
                    existing = self.kb.principle_collection.query(
                        query_texts=[p["summary"]],
                        n_results=1
                    )
                    
                    is_duplicate = False
                    if existing['documents'] and existing['distances'] and len(existing['distances'][0]) > 0:
                        # Chroma returns distance. Lower is closer.
                        # Threshold depends on metric. Default L2.
                        # Let's assume if distance is very small, it's a dupe.
                        # Actually, for MVP, let's just check exact string match or very high semantic match.
                        # Distance < 0.2 is usually very close for cosine/L2 on normalized vectors.
                        if existing['distances'][0][0] < 0.2: 
                            # Check if source is same, if so, definitely dupe.
                            # If source is different, maybe it's the same principle in another book?
                            # User said "Circle of competence" found 3 times.
                            # We will skip if similar principle exists.
                             is_duplicate = True
                    
                    if not is_duplicate:
                        p_id = str(uuid.uuid4())
                        p["id"] = p_id
                        p["source"] = source_name
                        self.kb.save_principle(p)
                        total_principles += 1
                
                # Save Cases
                for c in data.get("cases", []):
                    c_id = str(uuid.uuid4())
                    c["id"] = c_id
                    c["source"] = source_name
                    c["type"] = "Book Example"
                    self.kb.save_case(c)
                    total_cases += 1
                    
            except OpenAIError as e:
                # If it's a critical error (Auth or Quota), stop everything.
                if "insufficient_quota" in str(e) or "invalid_api_key" in str(e) or e.code in ["insufficient_quota", "invalid_api_key", 401, 429]:
                    raise e
                print(f"Error processing chunk {i}: {e}")
            except Exception as e:
                print(f"Error processing chunk {i}: {e}")
                
        return {"principles": total_principles, "cases": total_cases}

class DecisionEngine:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or "your_api_key_here" in api_key or "*****" in api_key:
            raise ValueError("Invalid OpenAI API Key. Please check your .env file.")
        self.client = OpenAI(api_key=api_key)

    def find_relevant_items(self, query: str, n_results: int = 3) -> Dict[str, List[Dict]]:
        # 1. Query Principles
        p_results = self.kb.principle_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # 2. Query Cases (Semantic Search)
        c_results = self.kb.case_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format results
        relevant_principles = []
        found_principle_summaries = []
        
        if p_results['documents']:
            for i, doc in enumerate(p_results['documents'][0]):
                # Try to extract the summary part (first line) for matching
                summary = doc.split('\n')[0]
                found_principle_summaries.append(summary)
                
                relevant_principles.append({
                    "content": doc,
                    "id": p_results['ids'][0][i],
                    "metadata": p_results['metadatas'][0][i]
                })

        relevant_cases = []
        
        # Add semantically similar cases
        if c_results['documents']:
            for i, doc in enumerate(c_results['documents'][0]):
                relevant_cases.append({
                    "content": doc,
                    "id": c_results['ids'][0][i],
                    "metadata": c_results['metadatas'][0][i],
                    "reason": "Semantic Match"
                })
        
        # 3. Find Linked Cases (Relational Search)
        # Look for cases that explicitly mention the found principles
        all_cases = self.kb.get_all_cases()
        for case in all_cases:
            related_p = case.get("related_principle_summary", "")
            # Simple fuzzy check: if one of the found principle summaries is in the case's related field
            for p_summary in found_principle_summaries:
                if p_summary in related_p or related_p in p_summary:
                    # Check if not already added
                    if not any(c['id'] == case['id'] for c in relevant_cases):
                        relevant_cases.append({
                            "content": f"{case['description']} (Linked to: {related_p})",
                            "id": case['id'],
                            "metadata": {"source": case.get("source"), "type": case.get("type")},
                            "reason": "Linked to Principle"
                        })
                    break
                
        return {"principles": relevant_principles, "cases": relevant_cases}

    def generate_advice(self, query: str, language: str = "English") -> Dict[str, Any]:
        context = self.find_relevant_items(query)
        
        context_str = "RELEVANT PRINCIPLES:\n"
        for p in context["principles"]:
            # Use Chinese content if available and language is Chinese
            content = p['content']
            if language == 'Chinese':
                # We need to fetch the full object to get the CN fields, 
                # but find_relevant_items returns a simplified dict.
                # For MVP, we'll just rely on the content string or try to parse.
                # Actually, let's just pass the English content for the LLM to understand, 
                # but instruct it to reply in Chinese.
                pass
            context_str += f"- {content}\n"
            
        context_str += "\nRELEVANT CASES:\n"
        for c in context["cases"]:
            context_str += f"- {c['content']}\n"

        system_prompt = f"""
        You are a wise decision consultant. Use the provided Principles and Cases to advise the user.
        
        Instructions:
        1. Analyze the user's problem.
        2. Cite specific principles that apply.
        3. Mention relevant cases if they are similar.
        4. Give actionable advice.
        5. **IMPORTANT**: Output your response in **{language}**.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context_str}\n\nUser Problem: {query}"}
            ]
        )
        
        advice_text = response.choices[0].message.content
        
        # Return structured data
        # We pass back the principles found by the search so the UI can link them.
        related_principles = []
        for p in context["principles"]:
            summary = p["metadata"].get("summary")
            if not summary:
                # Fallback: Extract from content (first line)
                summary = p["content"].split('\n')[0]
            related_principles.append(summary)

        return {
            "advice": advice_text,
            "related_principles": related_principles
        }
