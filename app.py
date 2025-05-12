from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import asyncio
import uvicorn

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
            messages=[message.model_dump() for message in request.messages]
        )
        return {"content": response.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

@app.post("/summary", response_model=SummaryResponse)
async def summary(request: SummaryRequest):
    async def get_summary(content):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "Summarize the following content into a very concise single sentence in Korean."},
                    {"role": "user", "content": content}
                ]
            )
        )
        return response.choices[0].message.content

    try:
        tasks = [get_summary(content) for content in request.contents]
        summaries = await asyncio.gather(*tasks)
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summaries: {str(e)}")

@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest):
    try:
        # Serialize the entire tree
        tree_json = request.tree.model_dump()
        tree_str = json.dumps(tree_json, ensure_ascii=False, indent=2)
        node_id = request.review_request
        # Prepare content for the prompt
        prompt = (
            "다음은 논증 구조를 트리 형태로 표현한 JSON입니다. "
            "각 노드는 id, type, content, summary, child, sibling 등의 정보를 포함합니다. "
            "아래 트리 전체를 참고하여, 주어진 nodeId(검토 대상 노드)에 대해 비판적 사고를 바탕으로 반론(반박) 또는 날카로운 질문을 한국어로 생성하세요. "
            "응답은 반드시 JSON 스키마에 맞춰주세요.\n"
            f"[트리 구조]:\n{tree_str}\n"
            f"[검토 대상 nodeId]: {node_id}"
        )
        print(prompt)
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "You are a critical thinker who can generate counterarguments and questions in Korean."},
                {"role": "user", "content": prompt}
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
            temperature=0.5,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            store=True
        )
        response_content = json.loads(response.choices[0].message.content)
        return {"data": response_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating review: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)