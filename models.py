from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Literal


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
    type: Literal["주제", "주장", "근거", "반론", "질문", "답변"]
    child: List["TreeNode"] = []
    sibling: List["TreeNode"] = []
    content: str
    summary: str
    created_by: str
    created_at: str
    updated_at: str


class ReviewRequest(BaseModel):
    tree: TreeNode
    review_num: int = 1  # Default to 1 if not provided
    student_id: str  # Student identifier
    assignment_id: str  # Assignment identifier


class EvidenceNode(BaseModel):
    type: Literal["근거"]
    content: str
    summary: str


class ReviewResponseTree(BaseModel):
    type: Literal["반론", "질문"]
    content: str
    summary: str
    child: Optional[List[EvidenceNode]] = None


class ReviewResponseData(BaseModel):
    parent: str
    tree: ReviewResponseTree


class ReviewResponse(BaseModel):
    data: List[ReviewResponseData]


# Models for /reset endpoint
class ResetResponse(BaseModel):
    message: str
    status: str


# Models for /resetall endpoint
class ResetAllResponse(BaseModel):
    message: str
    status: str
