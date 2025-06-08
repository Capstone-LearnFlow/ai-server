import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Cerebras client using AsyncOpenAI with Cerebras base URL
cerebras_client = AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=os.getenv("CEREBRAS_API_KEY"))

# Configuration constants
OPENAI_MODEL = "gpt-4.1-mini"
CEREBRAS_MODEL = "qwen-3-32b"
