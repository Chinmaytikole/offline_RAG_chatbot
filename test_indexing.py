import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Set up paths so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pageindex_service import PageIndexService

def test_indexing():
    api_key = os.getenv("PAGEINDEX_API_KEY")
    print(f"API Key read from env: {api_key}")
    
    service = PageIndexService(pageindex_api_key=api_key)
    
    print(f"Service available: {service.available}")
    if service.available:
        service.index_all_pdfs("uploads")
    
if __name__ == "__main__":
    test_indexing()
