from fastapi import APIRouter, HTTPException
from models import ChatRequest, ChatResponse
from services.openai_service import generate_chat_response

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests using OpenAI."""
    try:
        messages = [message.model_dump() for message in request.messages]
        content = await generate_chat_response(messages)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")
