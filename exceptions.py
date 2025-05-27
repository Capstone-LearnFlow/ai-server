from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with detailed logging."""
    print("=== 422 Validation Error Details ===")
    print(f"Request URL: {request.url}")
    print(f"Request Method: {request.method}")
    
    # Print detailed error information
    for i, error in enumerate(exc.errors()):
        print(f"Error {i + 1}:")
        print(f"  Location: {error['loc']}")
        print(f"  Message: {error['msg']}")
        print(f"  Type: {error['type']}")
        if 'input' in error:
            print(f"  Input: {error['input']}")
        if 'ctx' in error:
            print(f"  Context: {error['ctx']}")
        print()
    
    print("Raw request body:")
    try:
        body = await request.body()
        print(body.decode('utf-8'))
    except Exception as e:
        print(f"Could not read request body: {e}")
    print("=== End of Validation Error Details ===")
    
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )
