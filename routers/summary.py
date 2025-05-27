from fastapi import APIRouter, HTTPException
from models import SummaryRequest, SummaryResponse
from services.openai_service import generate_summaries

router = APIRouter()


@router.post("/summary", response_model=SummaryResponse)
async def summary(request: SummaryRequest):
    """Generate summaries for multiple content strings."""
    try:
        summaries = await generate_summaries(request.contents)
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summaries: {str(e)}")
