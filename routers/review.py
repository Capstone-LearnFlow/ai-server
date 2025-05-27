from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
import traceback
from models import ReviewRequest, ReviewResponse, ResetResponse
from services.review_service import ReviewService

router = APIRouter()

# Create a single instance of the review service to maintain state
review_service = ReviewService()


@router.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest):
    """Generate reviews (counterarguments or questions) for a tree structure."""
    try:
        ranked_reviews = await review_service.process_review_request(
            request.tree, 
            request.review_num
        )
        return {"data": ranked_reviews}
    except ValidationError as ve:
        print("=== Validation Error in /review endpoint ===")
        print(f"Validation Error: {ve}")
        print(f"Error details: {ve.errors()}")
        raise HTTPException(status_code=422, detail=f"Validation error: {ve.errors()}")
    except Exception as e:
        print(f"=== General Error in /review endpoint ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error generating review: {str(e)}")


@router.post("/reset", response_model=ResetResponse)
async def reset():
    """Reset the server state by clearing previous tree and unselected reviews."""
    try:
        review_service.reset_state()
        print("Server state has been reset successfully")
        
        return {
            "message": "Server state has been reset successfully",
            "status": "success"
        }
    except Exception as e:
        print(f"Error resetting server state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting server state: {str(e)}")
