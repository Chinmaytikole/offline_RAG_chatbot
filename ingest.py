# # ingest.py
# import os
# import shutil
# import json
# from pathlib import Path

# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain.schema import Document
# from langchain_community.embeddings import SentenceTransformerEmbeddings
# from sentence_transformers import SentenceTransformer
# from PIL import Image
# import fitz  # PyMuPDF

# class PDFIngestor:
#     def __init__(self, data_path="uploads", db_faiss_path="vector_store", image_dir="images"):
#         self.DATA_PATH = data_path
#         self.DB_FAISS_PATH = db_faiss_path
#         self.IMAGE_DIR = image_dir
#         self.MANIFEST_PATH = os.path.join(db_faiss_path, "manifest.json")
        
#         # Create necessary directories
#         os.makedirs(self.IMAGE_DIR, exist_ok=True)
#         os.makedirs(self.DB_FAISS_PATH, exist_ok=True)
        
#         # Initialize embedding model
#         self.clip_embeddings = SentenceTransformerEmbeddings(model_name="clip-ViT-B-32")

#     def get_current_files_state(self, directory=None):
#         """Get current state of PDF files in directory"""
#         if directory is None:
#             directory = self.DATA_PATH
#         return {str(filepath): os.path.getmtime(filepath) for filepath in Path(directory).rglob("*.pdf")}

#     def load_manifest(self):
#         """Load the manifest file"""
#         if os.path.exists(self.MANIFEST_PATH):
#             with open(self.MANIFEST_PATH, "r") as f:
#                 return json.load(f)
#         return {}

#     def save_manifest(self, state):
#         """Save the manifest file"""
#         os.makedirs(self.DB_FAISS_PATH, exist_ok=True)
#         with open(self.MANIFEST_PATH, "w") as f:
#             json.dump(state, f, indent=4)

#     def extract_text_around_image(self, page, image_bbox, margin=50):
#         """
#         Extract text surrounding an image within a specified margin.
#         This captures captions, labels, and contextual information.
#         """
#         try:
#             # Expand the image bbox to include surrounding area
#             expanded_bbox = (
#                 max(0, image_bbox[0] - margin),
#                 max(0, image_bbox[1] - margin),
#                 min(page.rect.width, image_bbox[2] + margin),
#                 min(page.rect.height, image_bbox[3] + margin)
#             )
            
#             # Extract text from the expanded area
#             text = page.get_text("text", clip=expanded_bbox).strip()
            
#             # If no text found in expanded area, try getting text from the entire page
#             # and look for text near the image coordinates
#             if not text:
#                 full_text = page.get_text()
#                 lines = full_text.split('\n')
                
#                 # Look for text lines that might be captions (above or below image)
#                 image_center_y = (image_bbox[1] + image_bbox[3]) / 2
#                 nearby_text = []
                
#                 for line in lines:
#                     if line.strip():
#                         # Simple heuristic: assume captions are short and meaningful
#                         words = line.split()
#                         if 2 <= len(words) <= 20:  # Captions are typically short
#                             nearby_text.append(line.strip())
                
#                 text = " | ".join(nearby_text[:3])  # Take top 3 potential captions
            
#             return text if text else "No contextual text found"
        
#         except Exception as e:
#             print(f"Error extracting text around image: {e}")
#             return "Error extracting contextual text"

#     def extract_images_from_pdf(self, pdf_path, output_folder=None):
#         """
#         Extract images from PDF with enhanced contextual information.
#         """
#         if output_folder is None:
#             output_folder = self.IMAGE_DIR
            
#         os.makedirs(output_folder, exist_ok=True)
#         doc = fitz.open(pdf_path)
#         image_info = []  # Store (path, description, metadata) tuples
        
#         print(f"📄 Processing {pdf_path} for images...")
        
#         for page_num, page in enumerate(doc):
#             image_list = page.get_images(full=True)
            
#             for img_index, img in enumerate(image_list):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 ext = base_image["ext"]
                
#                 # Create descriptive filename
#                 image_filename = f"{Path(pdf_path).stem}_p{page_num+1}_img{img_index+1}.{ext}"
#                 image_path = os.path.join(output_folder, image_filename)
                
#                 # Save image
#                 with open(image_path, "wb") as f:
#                     f.write(image_bytes)
                
#                 # Get image position on page
#                 image_bbox = None
#                 for image in page.get_images():
#                     if image[0] == xref:
#                         # Get the position where this image is displayed
#                         for xref_name in page.get_image_rects(xref):
#                             image_bbox = xref_name
#                             break
#                         break
                
#                 # Extract contextual text around the image
#                 contextual_text = ""
#                 if image_bbox:
#                     contextual_text = self.extract_text_around_image(page, image_bbox)
                
#                 # Get page text for additional context
#                 page_text = page.get_text().strip()
#                 page_context = page_text[:500] if page_text else ""  # First 500 chars for context
                
#                 # Create comprehensive description
#                 description_parts = []
                
#                 if contextual_text and contextual_text != "No contextual text found":
#                     description_parts.append(f"Context: {contextual_text}")
                
#                 if page_context:
#                     description_parts.append(f"Page content: {page_context}")
                
#                 # Fallback description if no good text found
#                 if not description_parts:
#                     description_parts.append(f"Image from {Path(pdf_path).stem} page {page_num+1}")
                
#                 description = ". ".join(description_parts)
                
#                 # Enhanced metadata
#                 metadata = {
#                     "source_pdf": pdf_path,
#                     "pdf_name": Path(pdf_path).stem,
#                     "page_number": page_num + 1,
#                     "image_index": img_index + 1,
#                     "contextual_text": contextual_text,
#                     "filename": image_filename,
#                     "image_type": ext.upper(),
#                     "has_context": bool(contextual_text and contextual_text != "No contextual text found")
#                 }
                
#                 image_info.append((image_path, description, metadata))
                
#                 print(f"  🖼️ Extracted image {img_index+1} on page {page_num+1}")
#                 if contextual_text and contextual_text != "No contextual text found":
#                     print(f"    📝 Context: {contextual_text[:100]}...")
        
#         doc.close()
#         return image_info

#     def full_rebuild(self):
#         """Perform a full rebuild of the vector store - ALWAYS deletes previous data"""
#         print("🔄 Performing FULL REBUILD - Deleting all previous embeddings...")

#         # Always clean up existing vector store and images
#         if os.path.exists(self.DB_FAISS_PATH):
#             shutil.rmtree(self.DB_FAISS_PATH)
#             print("🗑️ Removed existing vector store.")
        
#         # Clean up images directory
#         if os.path.exists(self.IMAGE_DIR):
#             shutil.rmtree(self.IMAGE_DIR)
#             print("🗑️ Removed existing images.")
        
#         # Recreate directories
#         os.makedirs(self.IMAGE_DIR, exist_ok=True)
#         os.makedirs(self.DB_FAISS_PATH, exist_ok=True)

#         # Check if there are any PDF files
#         pdf_files = list(Path(self.DATA_PATH).rglob("*.pdf"))
#         if not pdf_files:
#             print("❌ No PDF files found in uploads directory.")
#             return False

#         print(f"📚 Found {len(pdf_files)} PDF files. Loading documents...")
        
#         # Load PDFs
#         loader = DirectoryLoader(self.DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
#         documents = loader.load()
#         if not documents:
#             print("❌ No documents could be loaded from PDF files.")
#             return False

#         # Split text
#         print("✂️ Splitting text into chunks...")
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         text_chunks = splitter.split_documents(documents)
#         print(f"✅ Split {len(documents)} docs into {len(text_chunks)} chunks.")

#         all_docs = []

#         # Text documents
#         print("📝 Processing text chunks...")
#         for doc in text_chunks:
#             all_docs.append(Document(
#                 page_content=doc.page_content,
#                 metadata={
#                     "type": "text", 
#                     "source": doc.metadata.get("source", ""),
#                     "filename": os.path.basename(doc.metadata.get("source", "")),
#                     "content_type": "text_chunk"
#                 }
#             ))

#         # Image documents with enhanced descriptions and metadata
#         print("🖼️ Extracting images with contextual information...")
#         all_image_info = []
#         for pdf_path in Path(self.DATA_PATH).rglob("*.pdf"):
#             images_from_pdf = self.extract_images_from_pdf(str(pdf_path))
#             all_image_info.extend(images_from_pdf)
#             print(f"✅ Extracted {len(images_from_pdf)} images from {Path(pdf_path).name}")

#         print(f"📊 Total images extracted: {len(all_image_info)}")
        
#         # Count images with contextual information
#         images_with_context = sum(1 for _, _, metadata in all_image_info if metadata.get("has_context", False))
#         print(f"📝 Images with contextual text: {images_with_context}")

#         print("💾 Creating image documents for vector store...")
#         for image_path, description, metadata in all_image_info:
#             # Use the enhanced description for embedding
#             all_docs.append(Document(
#                 page_content=description,  # This gets embedded for semantic search
#                 metadata={
#                     "type": "image",
#                     "path": image_path,
#                     "filename": metadata["filename"],
#                     "source_pdf": metadata["source_pdf"],
#                     "pdf_name": metadata["pdf_name"],
#                     "page_number": metadata["page_number"],
#                     "image_index": metadata["image_index"],
#                     "contextual_text": metadata["contextual_text"],
#                     "image_type": metadata["image_type"],
#                     "has_context": metadata["has_context"],
#                     "content_type": "image_description"
#                 }
#             ))

#         # Build FAISS store
#         print("🏗️ Building FAISS vector store...")
#         vector_store = FAISS.from_documents(all_docs, embedding=self.clip_embeddings)
#         vector_store.save_local(self.DB_FAISS_PATH)

#         # Save manifest with current state
#         current_state = self.get_current_files_state()
#         self.save_manifest(current_state)
        
#         print("✅ FULL REBUILD complete!")
#         print(f"📊 Total documents in vector store: {len(all_docs)}")
#         print(f"📄 - Text chunks: {len(text_chunks)}")
#         print(f"🖼️ - Images: {len(all_image_info)}")
#         print(f"📝 - Images with contextual text: {images_with_context}")
        
#         return True

#     def run_ingestion(self):
#         """Run the complete ingestion process - ALWAYS does full rebuild"""
#         print("🚀 Starting PDF ingestion with enhanced image context extraction...")
#         print("⚠️  This will DELETE all previous embeddings and create new ones from scratch!")
        
#         # Check if uploads directory exists and has PDFs
#         if not os.path.exists(self.DATA_PATH):
#             print(f"❌ Uploads directory '{self.DATA_PATH}' not found.")
#             return False
            
#         pdf_files = list(Path(self.DATA_PATH).rglob("*.pdf"))
#         if not pdf_files:
#             print(f"❌ No PDF files found in '{self.DATA_PATH}' directory.")
#             print(f"💡 Please add PDF files to the '{self.DATA_PATH}' directory first.")
#             return False
        
#         print(f"📁 Found {len(pdf_files)} PDF files to process:")
#         for pdf_file in pdf_files:
#             print(f"   - {pdf_file}")
        
#         # ALWAYS do a full rebuild
#         success = self.full_rebuild()

#         if success:
#             print("🎉 Ingestion complete! All embeddings have been rebuilt from scratch.")
#         else:
#             print("❌ Ingestion failed.")
            
#         return success

# # Standalone execution
# if __name__ == "__main__":
#     ingestor = PDFIngestor()
#     ingestor.run_ingestion()
#     print("Run: uvicorn app:app --reload")




# ingest.py
import os
import shutil
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer
from PIL import Image
import fitz  # PyMuPDF

class PDFIngestor:
    def __init__(self, data_path="uploads", db_faiss_path="vector_store", image_dir="images"):
        self.DATA_PATH = data_path
        self.DB_FAISS_PATH = db_faiss_path
        self.IMAGE_DIR = image_dir
        self.MANIFEST_PATH = os.path.join(db_faiss_path, "manifest.json")
        
        # Create necessary directories
        os.makedirs(self.IMAGE_DIR, exist_ok=True)
        os.makedirs(self.DB_FAISS_PATH, exist_ok=True)
        
        # Initialize multilingual embedding model
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def get_current_files_state(self, directory=None):
        """Get current state of PDF files in directory"""
        if directory is None:
            directory = self.DATA_PATH
        return {str(filepath): os.path.getmtime(filepath) for filepath in Path(directory).rglob("*.pdf")}

    def load_manifest(self):
        """Load the manifest file"""
        if os.path.exists(self.MANIFEST_PATH):
            with open(self.MANIFEST_PATH, "r") as f:
                return json.load(f)
        return {}

    def save_manifest(self, state):
        """Save the manifest file"""
        os.makedirs(self.DB_FAISS_PATH, exist_ok=True)
        with open(self.MANIFEST_PATH, "w") as f:
            json.dump(state, f, indent=4)

    def extract_text_around_image(self, page, image_bbox, margin=50):
        """
        Extract text surrounding an image within a specified margin.
        This captures captions, labels, and contextual information.
        """
        try:
            # Expand the image bbox to include surrounding area
            expanded_bbox = (
                max(0, image_bbox[0] - margin),
                max(0, image_bbox[1] - margin),
                min(page.rect.width, image_bbox[2] + margin),
                min(page.rect.height, image_bbox[3] + margin)
            )
            
            # Extract text from the expanded area
            text = page.get_text("text", clip=expanded_bbox).strip()
            
            # If no text found in expanded area, try getting text from the entire page
            # and look for text near the image coordinates
            if not text:
                full_text = page.get_text()
                lines = full_text.split('\n')
                
                # Look for text lines that might be captions (above or below image)
                image_center_y = (image_bbox[1] + image_bbox[3]) / 2
                nearby_text = []
                
                for line in lines:
                    if line.strip():
                        # Simple heuristic: assume captions are short and meaningful
                        words = line.split()
                        if 2 <= len(words) <= 20:  # Captions are typically short
                            nearby_text.append(line.strip())
                
                text = " | ".join(nearby_text[:3])  # Take top 3 potential captions
            
            return text if text else "No contextual text found"
        
        except Exception as e:
            print(f"Error extracting text around image: {e}")
            return "Error extracting contextual text"

    def extract_images_from_pdf(self, pdf_path, output_folder=None):
        """
        Extract images from PDF with enhanced contextual information.
        """
        if output_folder is None:
            output_folder = self.IMAGE_DIR
            
        os.makedirs(output_folder, exist_ok=True)
        doc = fitz.open(pdf_path)
        image_info = []  # Store (path, description, metadata) tuples
        
        print(f"📄 Processing {pdf_path} for images...")
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                # Create descriptive filename
                image_filename = f"{Path(pdf_path).stem}_p{page_num+1}_img{img_index+1}.{ext}"
                image_path = os.path.join(output_folder, image_filename)
                
                # Save image
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Get image position on page
                image_bbox = None
                for image in page.get_images():
                    if image[0] == xref:
                        # Get the position where this image is displayed
                        for xref_name in page.get_image_rects(xref):
                            image_bbox = xref_name
                            break
                        break
                
                # Extract contextual text around the image
                contextual_text = ""
                if image_bbox:
                    contextual_text = self.extract_text_around_image(page, image_bbox)
                
                # Get page text for additional context
                page_text = page.get_text().strip()
                page_context = page_text[:500] if page_text else ""  # First 500 chars for context
                
                # Create comprehensive description
                description_parts = []
                
                if contextual_text and contextual_text != "No contextual text found":
                    description_parts.append(f"Context: {contextual_text}")
                
                if page_context:
                    description_parts.append(f"Page content: {page_context}")
                
                # Fallback description if no good text found
                if not description_parts:
                    description_parts.append(f"Image from {Path(pdf_path).stem} page {page_num+1}")
                
                description = ". ".join(description_parts)
                
                # Enhanced metadata
                metadata = {
                    "source_pdf": pdf_path,
                    "pdf_name": Path(pdf_path).stem,
                    "page_number": page_num + 1,
                    "image_index": img_index + 1,
                    "contextual_text": contextual_text,
                    "filename": image_filename,
                    "image_type": ext.upper(),
                    "has_context": bool(contextual_text and contextual_text != "No contextual text found")
                }
                
                image_info.append((image_path, description, metadata))
                
                print(f"  🖼️ Extracted image {img_index+1} on page {page_num+1}")
                if contextual_text and contextual_text != "No contextual text found":
                    print(f"    📝 Context: {contextual_text[:100]}...")
        
        doc.close()
        return image_info

    def full_rebuild(self):
        """Perform a full rebuild of the vector store - ALWAYS deletes previous data"""
        print("🔄 Performing FULL REBUILD - Deleting all previous embeddings...")

        # Always clean up existing vector store and images
        if os.path.exists(self.DB_FAISS_PATH):
            shutil.rmtree(self.DB_FAISS_PATH)
            print("🗑️ Removed existing vector store.")
        
        # Clean up images directory
        if os.path.exists(self.IMAGE_DIR):
            shutil.rmtree(self.IMAGE_DIR)
            print("🗑️ Removed existing images.")
        
        # Recreate directories
        os.makedirs(self.IMAGE_DIR, exist_ok=True)
        os.makedirs(self.DB_FAISS_PATH, exist_ok=True)

        # Check if there are any PDF files
        pdf_files = list(Path(self.DATA_PATH).rglob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found in uploads directory.")
            return False

        print(f"📚 Found {len(pdf_files)} PDF files. Loading documents...")
        
        # Load PDFs
        loader = DirectoryLoader(self.DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
        documents = loader.load()
        if not documents:
            print("❌ No documents could be loaded from PDF files.")
            return False

        # Split text
        print("✂️ Splitting text into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        text_chunks = splitter.split_documents(documents)
        print(f"✅ Split {len(documents)} docs into {len(text_chunks)} chunks.")

        all_docs = []

        # Text documents
        print("📝 Processing text chunks...")
        for doc in text_chunks:
            source_path = doc.metadata.get("source", "")
            all_docs.append(Document(
                page_content=doc.page_content,
                metadata={
                    "type": "text", 
                    "source": source_path,
                    "filename": os.path.basename(source_path),
                    "pdf_name": Path(source_path).stem,
                    "page_number": doc.metadata.get("page", 1),
                    "content_type": "text_chunk"
                }
            ))

        # Image documents with enhanced descriptions and metadata
        print("🖼️ Extracting images with contextual information...")
        all_image_info = []
        for pdf_path in Path(self.DATA_PATH).rglob("*.pdf"):
            images_from_pdf = self.extract_images_from_pdf(str(pdf_path))
            all_image_info.extend(images_from_pdf)
            print(f"✅ Extracted {len(images_from_pdf)} images from {Path(pdf_path).name}")

        print(f"📊 Total images extracted: {len(all_image_info)}")
        
        # Count images with contextual information
        images_with_context = sum(1 for _, _, metadata in all_image_info if metadata.get("has_context", False))
        print(f"📝 Images with contextual text: {images_with_context}")

        print("💾 Creating image documents for vector store...")
        for image_path, description, metadata in all_image_info:
            all_docs.append(Document(
                page_content=description,
                metadata={
                    "type": "image",
                    "path": image_path,
                    "filename": metadata["filename"],
                    "source_pdf": metadata["source_pdf"],
                    "pdf_name": metadata["pdf_name"],
                    "page_number": metadata["page_number"],
                    "image_index": metadata["image_index"],
                    "contextual_text": metadata["contextual_text"],
                    "image_type": metadata["image_type"],
                    "has_context": metadata["has_context"],
                    "content_type": "image_description"
                }
            ))

        # Build FAISS store with multilingual embeddings
        print("🏗️ Building FAISS vector store with multilingual embeddings...")
        vector_store = FAISS.from_documents(all_docs, embedding=self.embeddings)
        vector_store.save_local(self.DB_FAISS_PATH)

        # Save manifest with current state
        current_state = self.get_current_files_state()
        self.save_manifest(current_state)
        
        print("✅ FULL REBUILD complete!")
        print(f"📊 Total documents in vector store: {len(all_docs)}")
        print(f"📄 - Text chunks: {len(text_chunks)}")
        print(f"🖼️ - Images: {len(all_image_info)}")
        print(f"📝 - Images with contextual text: {images_with_context}")
        
        return True

    def run_ingestion(self):
        """Run the complete ingestion process - ALWAYS does full rebuild"""
        print("🚀 Starting PDF ingestion with enhanced image context extraction...")
        print("⚠️  This will DELETE all previous embeddings and create new ones from scratch!")
        
        # Check if uploads directory exists and has PDFs
        if not os.path.exists(self.DATA_PATH):
            print(f"❌ Uploads directory '{self.DATA_PATH}' not found.")
            return False 
            
        pdf_files = list(Path(self.DATA_PATH).rglob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in '{self.DATA_PATH}' directory.")
            print(f"💡 Please add PDF files to the '{self.DATA_PATH}' directory first.")
            return False
        
        print(f"📁 Found {len(pdf_files)} PDF files to process:")
        for pdf_file in pdf_files:
            print(f"   - {pdf_file}")
        
        # ALWAYS do a full rebuild
        success = self.full_rebuild()

        if success:
            print("🎉 Ingestion complete! All embeddings have been rebuilt from scratch.")
        else:
            print("❌ Ingestion failed.")
            
        return success

# Standalone execution
if __name__ == "__main__":
    ingestor = PDFIngestor()
    ingestor.run_ingestion()
    print("Run: uvicorn app:app --reload")