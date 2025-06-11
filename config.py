import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Perplexity API client
perplexity_client = AsyncOpenAI(base_url="https://api.perplexity.ai", api_key=os.getenv("PPLX_API_KEY"))

# Initialize Cerebras client using AsyncOpenAI with Cerebras base URL
cerebras_client = AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=os.getenv("CEREBRAS_API_KEY"))

# Configuration constants
OPENAI_MODEL = "gpt-4.1"
OPENAI_MINI_MODEL = "gpt-4.1-mini"
PERPLEXITY_MODEL = "llama-3-70b-instruct"  # Updated to a supported Perplexity model
CEREBRAS_MODEL = "qwen-3-32b"
