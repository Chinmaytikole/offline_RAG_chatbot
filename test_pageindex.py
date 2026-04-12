import os
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

api_key = os.getenv("PAGEINDEX_API_KEY")
print(f"API Key: {api_key}")

try:
    client = PageIndexClient(api_key=api_key)
    print("✅ Client initialized")
    
    # Try a simple call
    docs = client.list_documents()
    print(f"✅ Documents in account: {len(docs)}")
except Exception as e:
    print(f"❌ Error: {e}")
