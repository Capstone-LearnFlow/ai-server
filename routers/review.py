from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError
import traceback
from models import ReviewRequest, ReviewResponse, ResetResponse, ResetAllResponse
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
            request.review_num,
            request.student_id,
            request.assignment_id
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
async def reset(student_id: str = Query(..., description="Student identifier"), 
                assignment_id: str = Query(..., description="Assignment identifier")):
    """Reset the server state for a specific student and assignment."""
    try:
        review_service.reset_state(student_id, assignment_id)
        message = f"Server state for student {student_id} and assignment {assignment_id} has been reset successfully"
        print(message)
        
        return {
            "message": message,
            "status": "success"
        }
    except Exception as e:
        print(f"Error resetting server state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting server state: {str(e)}")


@router.post("/resetall", response_model=ResetAllResponse)
async def reset_all():
    """Reset all server states by clearing all stored data."""
    try:
        review_service.reset_all()
        message = "All server states have been reset successfully"
        print(message)
        
        return {
            "message": message,
            "status": "success"
        }
    except Exception as e:
        print(f"Error resetting all server states: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting all server states: {str(e)}")
