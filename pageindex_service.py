"""
PageIndex (Vectorless RAG) Service
===================================
Handles document tree indexing via the PageIndex SDK and LLM-based
tree-search to retrieve exact sections without vector similarity.

Key Concept:
  Traditional RAG → chunk → embed → cosine similarity → retrieve
  PageIndex RAG   → build tree → LLM reasons over tree → retrieve exact sections
"""

import os
import json
import time
import threading
from typing import Optional

# --- Lazy imports (graceful if packages are missing) ---
try:
    from pageindex import PageIndexClient
    PAGEINDEX_SDK_AVAILABLE = True
except ImportError:
    PAGEINDEX_SDK_AVAILABLE = False

try:
    from openai import OpenAI as OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Manifest helpers – persists doc_id ↔ pdf_path mappings locally
# ---------------------------------------------------------------------------
PAGEINDEX_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pageindex_data")
MANIFEST_PATH = os.path.join(PAGEINDEX_DATA_DIR, "manifest.json")


def _load_manifest() -> dict:
    """Load the local manifest that maps PDF paths → PageIndex doc metadata."""
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(manifest: dict):
    os.makedirs(PAGEINDEX_DATA_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# PageIndexService
# ---------------------------------------------------------------------------
class PageIndexService:
    """Manages PageIndex document indexing + tree-based LLM retrieval."""

    def __init__(
        self,
        pageindex_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o",
        openai_base_url: Optional[str] = None,
    ):
        self.available = False
        self._lock = threading.Lock()

        # PageIndex client
        if PAGEINDEX_SDK_AVAILABLE and pageindex_api_key:
            try:
                self.pi_client = PageIndexClient(api_key=pageindex_api_key)
                self._pi_key = pageindex_api_key
                self.available = True
                print("✅ PageIndex client initialized")
            except Exception as e:
                print(f"⚠️ PageIndex client init failed: {e}")
                self.pi_client = None
        else:
            self.pi_client = None
            if not PAGEINDEX_SDK_AVAILABLE:
                print("⚠️ PageIndex SDK not installed. Install with: pip install pageindex")
            elif not pageindex_api_key:
                print("⚠️ PAGEINDEX_API_KEY not set – vectorless RAG disabled")

        # OpenAI client (for the tree-search LLM reasoning step)
        if OPENAI_AVAILABLE and (openai_api_key or openai_base_url):
            try:
                # pass base_url for Jan AI compatibility
                kwargs = {}
                # dummy api key can be used if base_url is set and no key is provided
                kwargs["api_key"] = openai_api_key or "dummy" 
                if openai_base_url:
                    kwargs["base_url"] = openai_base_url
                self.openai_client = OpenAIClient(**kwargs)
                self._openai_model = openai_model
                print(f"✅ OpenAI client initialized (model: {openai_model}, target: {openai_base_url or 'OpenAI Native'})")
            except Exception as e:
                print(f"⚠️ OpenAI client init failed: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
            if not OPENAI_AVAILABLE:
                print("⚠️ openai package not installed for tree-search LLM")
            elif not (openai_api_key or openai_base_url):
                print("⚠️ Neither OPENAI_API_KEY nor base_url set – tree-search LLM disabled")

        # Load local manifest
        self.manifest = _load_manifest()

    # ------------------------------------------------------------------
    # Document indexing
    # ------------------------------------------------------------------

    def submit_and_index(self, pdf_path: str) -> Optional[str]:
        """
        Submit a PDF to PageIndex for tree indexing.
        Returns the doc_id if successful, or None on failure.
        Caches in the manifest so we don't re-upload the same file.
        """
        if not self.available or not self.pi_client:
            return None

        abs_path = os.path.abspath(pdf_path)

        # Already indexed?
        if abs_path in self.manifest:
            entry = self.manifest[abs_path]
            if entry.get("status") == "completed":
                print(f"📄 Already indexed: {os.path.basename(pdf_path)}")
                return entry["doc_id"]

        print(f"📤 Submitting to PageIndex: {os.path.basename(pdf_path)} ...")
        try:
            result = self.pi_client.submit_document(pdf_path)
            doc_id = result.get("doc_id") or result.get("id")
            if not doc_id:
                print(f"⚠️ PageIndex returned no doc_id for {pdf_path}")
                return None

            # Poll until processing completes (max ~5 minutes)
            status = "processing"
            for _ in range(60):
                info = self.pi_client.get_document(doc_id)
                status = info.get("status", "processing")
                if status == "completed":
                    break
                if status in ("failed", "error"):
                    print(f"❌ PageIndex indexing failed for {pdf_path}: {info}")
                    self.manifest[abs_path] = {
                        "doc_id": doc_id,
                        "status": "failed",
                        "filename": os.path.basename(pdf_path),
                    }
                    _save_manifest(self.manifest)
                    return None
                time.sleep(5)

            if status != "completed":
                print(f"⏰ PageIndex indexing timed out for {pdf_path}")
                self.manifest[abs_path] = {
                    "doc_id": doc_id,
                    "status": "timeout",
                    "filename": os.path.basename(pdf_path),
                }
                _save_manifest(self.manifest)
                return None

            # Fetch and cache the tree
            tree = self._fetch_tree(doc_id)

            with self._lock:
                self.manifest[abs_path] = {
                    "doc_id": doc_id,
                    "status": "completed",
                    "filename": os.path.basename(pdf_path),
                    "tree": tree,
                }
                _save_manifest(self.manifest)

            print(f"✅ PageIndex indexing completed: {os.path.basename(pdf_path)}")
            return doc_id

        except Exception as e:
            print(f"❌ PageIndex submit error for {pdf_path}: {e}")
            return None

    def _fetch_tree(self, doc_id: str) -> dict:
        """Fetch the hierarchical tree structure for a document."""
        try:
            doc_info = self.pi_client.get_document(doc_id)
            return doc_info.get("tree", doc_info.get("index", {}))
        except Exception as e:
            print(f"⚠️ Could not fetch tree for {doc_id}: {e}")
            return {}

    def index_all_pdfs(self, uploads_dir: str):
        """Submit all PDFs in the uploads directory for PageIndex indexing."""
        if not self.available:
            print("⚠️ PageIndex service not available. Skipping automatic indexing.")
            return

        pdf_files = []
        for root, _dirs, files in os.walk(uploads_dir):
            for fname in files:
                if fname.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, fname))
        
        if not pdf_files:
            print(f"ℹ️ No PDFs found in {uploads_dir} for PageIndex.")
            return

        print(f"🔍 Found {len(pdf_files)} PDFs in {uploads_dir}. Checking PageIndex status...")
        for pdf_path in pdf_files:
            self.submit_and_index(pdf_path)
        print("✅ PageIndex automatic indexing check complete.")

    def remove_document(self, pdf_path: str):
        """Remove a document from the local manifest."""
        abs_path = os.path.abspath(pdf_path)
        if abs_path in self.manifest:
            with self._lock:
                del self.manifest[abs_path]
                _save_manifest(self.manifest)
            print(f"🗑️ Removed PageIndex entry for {os.path.basename(pdf_path)}")

    # ------------------------------------------------------------------
    # Tree-search via LLM
    # ------------------------------------------------------------------

    def _compress_tree(self, tree: dict, max_depth: int = 4, _depth: int = 0) -> str:
        """Compress tree to a readable outline for the LLM."""
        if not tree:
            return ""

        lines = []
        indent = "  " * _depth
        title = tree.get("title") or tree.get("name") or tree.get("heading", "")
        node_id = tree.get("id") or tree.get("node_id", "")
        page = tree.get("page", "")

        if title:
            page_info = f" (p.{page})" if page else ""
            lines.append(f"{indent}- [{node_id}] {title}{page_info}")

        if _depth < max_depth:
            for child in tree.get("children", []):
                lines.append(self._compress_tree(child, max_depth, _depth + 1))

        return "\n".join(line for line in lines if line)

    def llm_tree_search(self, query: str, trees: list[dict]) -> dict:
        """
        Use the LLM to reason over document trees and identify
        the most relevant sections (nodes) for the query.

        Returns: {"thinking": str, "node_ids": list[str], "sections": list[str]}
        """
        if not self.openai_client:
            return {"thinking": "", "node_ids": [], "sections": []}

        # Build the tree outline
        tree_text = ""
        for i, entry in enumerate(trees):
            tree = entry.get("tree", {})
            fname = entry.get("filename", f"Document {i+1}")
            outline = self._compress_tree(tree)
            if outline:
                tree_text += f"\n## {fname}\n{outline}\n"

        if not tree_text.strip():
            return {"thinking": "No tree structure available", "node_ids": [], "sections": []}

        prompt = f"""You are a document retrieval assistant. You have a hierarchical table-of-contents (tree) for one or more documents. Each node has an ID in square brackets.

Given a user's question, identify the TOP 3 most relevant sections by their node IDs. Think step-by-step about which sections would contain the answer.

DOCUMENT TREE:
{tree_text}

USER QUESTION: {query}

Respond ONLY in this JSON format:
{{
  "thinking": "your step-by-step reasoning about which sections are relevant",
  "node_ids": ["id1", "id2", "id3"],
  "section_titles": ["title1", "title2", "title3"]
}}"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON from the response (handle markdown code fences)
            json_str = raw
            if "```" in json_str:
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
                json_str = json_str.strip()

            result = json.loads(json_str)
            return {
                "thinking": result.get("thinking", ""),
                "node_ids": result.get("node_ids", []),
                "sections": result.get("section_titles", []),
            }

        except Exception as e:
            print(f"⚠️ LLM tree-search error: {e}")
            return {"thinking": f"Error: {e}", "node_ids": [], "sections": []}

    def retrieve_content_for_nodes(self, doc_id: str, node_ids: list[str]) -> str:
        """Retrieve the actual text content for the given node IDs from PageIndex."""
        if not self.pi_client or not node_ids:
            return ""

        try:
            contents = []
            for node_id in node_ids[:5]:  # Limit to 5 nodes
                try:
                    chunk = self.pi_client.get_chunk(doc_id, node_id)
                    text = chunk.get("content") or chunk.get("text", "")
                    if text:
                        contents.append(text)
                except Exception:
                    continue
            return "\n\n---\n\n".join(contents)
        except Exception as e:
            print(f"⚠️ Error retrieving node content: {e}")
            return ""

    # ------------------------------------------------------------------
    # High-level: build an answer from PageIndex
    # ------------------------------------------------------------------

    def build_answer(self, query: str, language: str = "en") -> Optional[dict]:
        """
        End-to-end PageIndex answer generation:
        1. Gather all indexed document trees
        2. LLM reasons over trees to find relevant sections
        3. Retrieve the actual content for those sections
        4. Generate a final answer using the LLM

        Returns dict with keys: answer, thinking, sections
        """
        if not self.available or not self.openai_client:
            return None

        # Collect all completed document trees
        indexed_docs = []
        for path, entry in self.manifest.items():
            if entry.get("status") == "completed" and entry.get("tree"):
                indexed_docs.append(entry)

        if not indexed_docs:
            return None

        # Step 1: LLM tree search
        search_result = self.llm_tree_search(query, indexed_docs)
        thinking = search_result.get("thinking", "")
        node_ids = search_result.get("node_ids", [])
        sections = search_result.get("sections", [])

        # Step 2: Retrieve actual content for matched nodes
        retrieved_content = ""
        for entry in indexed_docs:
            doc_id = entry.get("doc_id")
            if doc_id and node_ids:
                content = self.retrieve_content_for_nodes(doc_id, node_ids)
                if content:
                    retrieved_content += f"\n\n{content}"

        # Step 3: Generate answer using retrieved content
        if not retrieved_content.strip():
            # Fallback: if we couldn't retrieve content via nodes, use section titles
            if sections:
                retrieved_content = (
                    "The most relevant sections identified are:\n"
                    + "\n".join(f"- {s}" for s in sections)
                    + "\n\n(Detailed content retrieval was not available.)"
                )
            else:
                return {
                    "answer": "PageIndex could not find relevant sections for this query.",
                    "thinking": thinking,
                    "sections": sections,
                }

        answer = self._generate_answer(query, retrieved_content, language)

        return {
            "answer": answer,
            "thinking": thinking,
            "sections": sections,
        }

    def _generate_answer(self, query: str, context: str, language: str = "en") -> str:
        """Generate a final answer using the OpenAI LLM based on retrieved context."""
        lang_instructions = {
            "en": "Answer in English.",
            "es": "Responde en español.",
            "fr": "Répondez en français.",
            "de": "Antworten Sie auf Deutsch.",
            "hi": "हिन्दी में उत्तर दें।",
            "zh": "请用中文回答。",
            "ja": "日本語で回答してください。",
            "ko": "한국어로 대답하세요.",
        }
        lang_note = lang_instructions.get(language, "Answer in the same language as the question.")

        prompt = f"""{lang_note}

Based on the following content retrieved from the document tree structure,
provide a clear, accurate, and concise answer to the question.

RETRIEVED CONTENT:
{context[:3000]}

QUESTION: {query}

ANSWER:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating PageIndex answer: {e}"

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return the indexing status for all tracked documents."""
        docs = []
        for path, entry in self.manifest.items():
            docs.append({
                "path": path,
                "filename": entry.get("filename", os.path.basename(path)),
                "status": entry.get("status", "unknown"),
                "doc_id": entry.get("doc_id", ""),
                "has_tree": bool(entry.get("tree")),
            })
        return {
            "available": self.available,
            "openai_configured": self.openai_client is not None,
            "total_documents": len(docs),
            "documents": docs,
        }
