import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
client = AsyncOpenAI(base_url=os.getenv("LITELLM_BASE_URL"), api_key=os.getenv("LITELLM_API_KEY"))

# Initialize Perplexity API client
perplexity_client = AsyncOpenAI(base_url="https://api.perplexity.ai", api_key=os.getenv("PPLX_API_KEY"))

# Initialize Cerebras client using AsyncOpenAI with Cerebras base URL
cerebras_client = AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=os.getenv("CEREBRAS_API_KEY"))

# Configuration constants
OPENAI_MODEL = "gpt-4.1"
OPENAI_MINI_MODEL = "gpt-4.1-mini"
PERPLEXITY_MODEL = "sonar" 
CEREBRAS_MODEL = "qwen-3-32b"
