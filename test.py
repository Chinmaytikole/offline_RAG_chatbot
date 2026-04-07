#!/usr/bin/env python3
"""
Standalone VectorDB Test Script
Run this from command line to test your vector database queries
"""

import os
import sys
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append('.')

def test_vectordb():
    """Test function to query vector database and show results"""
    
    try:
        # Import required libraries
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        import numpy as np
        
        print("🔍 VectorDB Test Script")
        print("=" * 50)
        
        # Configuration
        DB_FAISS_PATH = "vector_store"
        MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        # Check if vector store exists
        if not os.path.exists(DB_FAISS_PATH) or not os.listdir(DB_FAISS_PATH):
            print("❌ Vector store not found or empty!")
            print("💡 Please run the ingestion process first.")
            return
        
        print(f"✅ Vector store found at: {DB_FAISS_PATH}")
        
        # Load embeddings
        print("🔄 Loading embeddings model...")
        embeddings = SentenceTransformerEmbeddings(model_name=MODEL_NAME)
        
        # Load vector store
        print("🔄 Loading vector database...")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector database loaded successfully!")
        
        # Test query loop
        print("\n🎯 Ready for queries. Type 'quit' to exit.")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                query = input("\n🔍 Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not query:
                    print("⚠️  Please enter a query")
                    continue
                
                # Get k value
                try:
                    k = int(input("📊 Number of results to show (default 5): ") or "5")
                except ValueError:
                    k = 5
                
                print(f"\n🔄 Searching for: '{query}' (showing top {k} results)")
                print("=" * 60)
                
                # Method 1: Regular similarity search
                print("\n1. 📖 Regular Similarity Search:")
                print("-" * 40)
                docs = db.similarity_search(query, k=k)
                
                for i, doc in enumerate(docs, 1):
                    print(f"\n📄 Result {i}:")
                    print(f"   Content: {doc.page_content[:200]}..." if len(doc.page_content) > 200 else f"   Content: {doc.page_content}")
                    print(f"   Metadata: {json.dumps(doc.metadata, indent=6, default=str)}")
                    print(f"   Content Length: {len(doc.page_content)} chars")
                    print(f"   Score: Similarity search (no direct score)")
                
                # Method 2: Similarity search with scores
                print(f"\n2. 🎯 Similarity Search with Scores:")
                print("-" * 40)
                docs_with_scores = db.similarity_search_with_score(query, k=k)
                
                for i, (doc, score) in enumerate(docs_with_scores, 1):
                    print(f"\n📄 Result {i} (Score: {score:.4f}):")
                    print(f"   Content: {doc.page_content[:150]}..." if len(doc.page_content) > 150 else f"   Content: {doc.page_content}")
                    print(f"   Type: {doc.metadata.get('type', 'unknown')}")
                    print(f"   Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"   Filename: {doc.metadata.get('filename', 'N/A')}")
                    if doc.metadata.get('page_number'):
                        print(f"   Page: {doc.metadata.get('page_number')}")
                    if doc.metadata.get('content_type'):
                        print(f"   Content Type: {doc.metadata.get('content_type')}")
                
                # Method 3: Search by vector
                print(f"\n3. 🔧 Vector-based Search:")
                print("-" * 40)
                query_embedding = embeddings.embed_query(query)
                vector_docs = db.similarity_search_by_vector(np.array(query_embedding), k=k)
                
                for i, doc in enumerate(vector_docs, 1):
                    print(f"\n📄 Result {i}:")
                    print(f"   Content: {doc.page_content[:100]}..." if len(doc.page_content) > 100 else f"   Content: {doc.page_content}")
                    print(f"   Metadata Keys: {list(doc.metadata.keys())}")
                
                # Statistics
                print(f"\n4. 📊 Search Statistics:")
                print("-" * 40)
                total_docs = len(docs)
                text_docs = sum(1 for doc in docs if doc.metadata.get('type') == 'text')
                image_docs = sum(1 for doc in docs if doc.metadata.get('type') == 'image')
                
                print(f"   Total results: {total_docs}")
                print(f"   Text chunks: {text_docs}")
                print(f"   Image descriptions: {image_docs}")
                
                # Show unique sources
                sources = set()
                for doc in docs:
                    source = doc.metadata.get('source', 'Unknown')
                    if source != 'Unknown':
                        sources.add(os.path.basename(source))
                
                if sources:
                    print(f"   Sources: {', '.join(list(sources)[:3])}{'...' if len(sources) > 3 else ''}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error during query: {e}")
                continue
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you have all required packages installed:")
        print("   pip install langchain-community faiss-cpu sentence-transformers numpy")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your vector store exists and is properly configured")

def quick_test():
    """Quick test function for basic verification"""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        
        print("🚀 Quick VectorDB Test")
        print("=" * 30)
        
        DB_FAISS_PATH = "vector_store"
        
        if not os.path.exists(DB_FAISS_PATH):
            print("❌ Vector store not found!")
            return False
        
        # Load database
        embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
        
        # Test query
        test_query = "artificial intelligence"
        docs = db.similarity_search(test_query, k=3)
        
        print(f"✅ VectorDB loaded successfully!")
        print(f"📊 Total documents in index: {db.index.ntotal}")
        print(f"🔍 Test query '{test_query}' returned {len(docs)} results")
        
        if docs:
            print(f"📄 First result type: {docs[0].metadata.get('type', 'unknown')}")
            print(f"📄 First result source: {os.path.basename(docs[0].metadata.get('source', 'unknown'))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

if __name__ == "__main__":
    print("VectorDB Test Script")
    print("====================")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_test()
        elif sys.argv[1] == "query" and len(sys.argv) > 2:
            # Run a single query from command line
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_community.embeddings import SentenceTransformerEmbeddings
                
                DB_FAISS_PATH = "vector_store"
                embeddings = SentenceTransformerEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                db = FAISS.load_local(DB_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
                
                query = " ".join(sys.argv[2:])
                k = 5
                
                print(f"🔍 Query: {query}")
                docs = db.similarity_search_with_score(query, k=k)
                
                for i, (doc, score) in enumerate(docs, 1):
                    print(f"\n{i}. Score: {score:.4f}")
                    print(f"   Type: {doc.metadata.get('type', 'unknown')}")
                    print(f"   Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}")
                    print(f"   Content: {doc.page_content[:100]}...")
                    
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("Usage:")
            print("  python test_vectordb.py              # Interactive mode")
            print("  python test_vectordb.py quick        # Quick test")
            print("  python test_vectordb.py query <text> # Single query")
    else:
        # Run interactive mode
        test_vectordb()