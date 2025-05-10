from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="OpenAI API Integration Server")

# Models for /chat endpoint
class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "gpt-4.1-mini"

class ChatResponse(BaseModel):
    content: str

# Models for /summary endpoint
class SummaryRequest(BaseModel):
    contents: List[str]

class SummaryResponse(BaseModel):
    summaries: List[str]

# Models for /review endpoint
class TreeNode(BaseModel):
    id: str
    type: Literal["subject", "claim", "reason", "cargument", "question"]
    child: List["TreeNode"] = []
    sibling: List["TreeNode"] = []
    content: str
    summary: str
    created_by: str
    created_at: str
    updated_at: str

class ReviewRequest(BaseModel):
    tree: TreeNode
    review_request: str  # nodeId

class ReviewResponseTree(BaseModel):
    type: Literal["반론", "질문"]
    content: str
    summary: str

class ReviewResponseData(BaseModel):
    parent: str
    tree: ReviewResponseTree

class ReviewResponse(BaseModel):
    data: ReviewResponseData

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[message.dict() for message in request.messages]
        )
        return {"content": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

@app.post("/summary", response_model=SummaryResponse)
async def summary(request: SummaryRequest):
    try:
        summaries = []
        for content in request.contents:
            prompt = f"Please summarize the following content into a very concise single sentence: {content}"
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides very concise one-line summaries."},
                    {"role": "user", "content": prompt}
                ]
            )
            summaries.append(response.choices[0].message.content)
        
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summaries: {str(e)}")

@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest):
    try:
        # Find the node with the requested ID
        def find_node(node, node_id):
            if node.id == node_id:
                return node
            
            for child in node.child:
                result = find_node(child, node_id)
                if result:
                    return result
            
            for sibling in node.sibling:
                result = find_node(sibling, node_id)
                if result:
                    return result
            
            return None
        
        target_node = find_node(request.tree, request.review_request)
        if not target_node:
            raise HTTPException(status_code=404, detail="Node not found")
        
        # Prepare content for the prompt
        node_content = target_node.content
        node_context = f"Node type: {target_node.type}, Content: {node_content}"
        
        # Generate counterargument or question using OpenAI
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a critical thinker who can generate counterarguments and questions."},
                {"role": "user", "content": f"Please review this statement and generate either a counterargument or a thought-provoking question: {node_context}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "parent_node_argument_question",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "parent": {
                                "type": "string",
                                "description": "The identifier of the parent node."
                            },
                            "tree": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "description": "Indicates whether the node is an argument (반론) or a question (질문).",
                                        "enum": ["반론", "질문"]
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The content of the argument or question."
                                    },
                                    "summary": {
                                        "type": "string",
                                        "description": "A brief summary of the argument or question."
                                    }
                                },
                                "required": ["type", "content", "summary"],
                                "additionalProperties": False
                            }
                        },
                        "required": ["parent", "tree"],
                        "additionalProperties": False
                    }
                }
            },
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        # Extract and return the response
        return {"data": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating review: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)