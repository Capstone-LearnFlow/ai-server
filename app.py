import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from exceptions import validation_exception_handler
from routers import chat, summary, review

app = FastAPI(title="OpenAI API Integration Server")

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)

# Include routers
app.include_router(chat.router)
app.include_router(summary.router)
app.include_router(review.router)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)