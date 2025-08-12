"""Example of using the GenAI client through BM singleton.

This example demonstrates how to access the Google GenAI client
configured with Vertex AI through the BM singleton.
"""

from buttermilk._core.dmrc import get_bm

# Get the BM singleton instance
bm = get_bm()

# Access the GenAI client
# Note: This requires proper GCP configuration with project_id and location
try:
    genai_client = bm.genai
    print(f"Successfully got GenAI client: {genai_client}")
    
    # Now you can use the client for various GenAI operations
    # For example, listing models available in your project:
    # models = genai_client.models.list()
    
except RuntimeError as e:
    print(f"Error getting GenAI client: {e}")
    print("Make sure your GCP configuration includes:")
    print("  - project_id (or project)")
    print("  - location")