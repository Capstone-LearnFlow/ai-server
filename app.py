from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional, Union
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import asyncio
import uvicorn
from copy import deepcopy

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
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

# In-memory storage for previous tree
previous_tree = None

def find_node_by_id(tree: TreeNode, node_id: str) -> Optional[TreeNode]:
    """Find a node in the tree by its ID."""
    if tree.id == node_id:
        return tree
    
    # Search in child nodes
    for child in tree.child:
        result = find_node_by_id(child, node_id)
        if result:
            return result
    
    # Search in sibling nodes
    for sibling in tree.sibling:
        result = find_node_by_id(sibling, node_id)
        if result:
            return result
    
    return None

def get_all_nodes(tree: TreeNode) -> Dict[str, TreeNode]:
    """Get all nodes in the tree as a dictionary with node IDs as keys."""
    result = {tree.id: tree}
    
    # Process child nodes
    for child in tree.child:
        result.update(get_all_nodes(child))
    
    # Process sibling nodes
    for sibling in tree.sibling:
        result.update(get_all_nodes(sibling))
    
    return result

def find_new_nodes(current_tree: Dict[str, TreeNode], previous_tree: Dict[str, TreeNode]) -> List[TreeNode]:
    """Find nodes that were added since the previous tree state."""
    new_nodes = []
    
    for node_id, node in current_tree.items():
        # If node didn't exist before or is of type 근거 or 반론 and has been updated
        if (node_id not in previous_tree or 
            (node.type in ["근거", "반론"] and 
             (node.updated_at != previous_tree[node_id].updated_at if node_id in previous_tree else True))):
            new_nodes.append(node)
    
    return new_nodes

async def generate_review(node: TreeNode, tree: TreeNode) -> Dict[str, Any]:
    """Generate a review (counterargument or question) for a given node."""
    tree_json = tree.model_dump()
    tree_str = json.dumps(tree_json, ensure_ascii=False, indent=2)
    
    prompt = (
        "다음은 논증 구조를 트리 형태로 표현한 JSON입니다. "
        "각 노드는 id, type, content, summary, child, sibling 등의 정보를 포함합니다. "
        "아래 트리 전체를 참고하여, 주어진 nodeId(검토 대상 노드)에 대해 비판적 사고를 바탕으로 반론(반박) 또는 날카로운 질문을 한국어로 생성하세요. "
        "IMPORTANT: 반론에는 반드시 child에 type이 근거인 노드가 포함되어야 하며, 질문은 명확하고 구체적이어야 합니다.\n"
        "응답은 반드시 JSON 스키마에 맞춰주세요.\n"
        f"[트리 구조]:\n{tree_str}\n"
        f"[검토 대상 nodeId]: {node.id}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
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
                            "anyOf": [
                                {
                                    # 반론 타입일 경우의 스키마
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "const": "반론",
                                            "description": "Indicates this is an argument node."
                                        },
                                        "content": {
                                            "type": "string",
                                            "description": "The content of the argument."
                                        },
                                        "summary": {
                                            "type": "string",
                                            "description": "A brief summary of the argument."
                                        },
                                        "child": {
                                            "type": "array",
                                            "description": "Evidence supporting the argument. IMPORTANT: 한개 이상의 근거를 반드시 포함하세요.",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string",
                                                        "const": "근거",
                                                        "description": "Indicates this is an evidence node."
                                                    },
                                                    "content": {
                                                        "type": "string",
                                                        "description": "The content of the evidence."
                                                    },
                                                    "summary": {
                                                        "type": "string",
                                                        "description": "A brief summary of the evidence."
                                                    }
                                                },
                                                "required": ["type", "content", "summary"],
                                                "additionalProperties": False
                                            }
                                        }
                                    },
                                    "required": ["type", "content", "summary", "child"],
                                    "additionalProperties": False
                                },
                                {
                                    # 질문 타입일 경우의 스키마
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "const": "질문",
                                            "description": "Indicates this is a question node."
                                        },
                                        "content": {
                                            "type": "string",
                                            "description": "The content of the question."
                                        },
                                        "summary": {
                                            "type": "string",
                                            "description": "A brief summary of the question."
                                        }
                                    },
                                    "required": ["type", "content", "summary"],
                                    "additionalProperties": False
                                }
                            ]
                        }
                    },
                    "required": ["parent", "tree"],
                    "additionalProperties": False
                }
            }
        },
        temperature=0.7,  # Use slightly higher temperature for diversity
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        store=True
    )
    
    return json.loads(response.choices[0].message.content)

async def rank_reviews(reviews: List[Dict[str, Any]], tree: TreeNode, review_num: int) -> List[Dict[str, Any]]:
    """Rank the generated reviews and return the top ones."""
    if len(reviews) <= review_num:
        return reviews
    
    tree_json = tree.model_dump()
    tree_str = json.dumps(tree_json, ensure_ascii=False, indent=2)
    reviews_str = json.dumps(reviews, ensure_ascii=False, indent=2)
    
    prompt = (
        "다음은 논증 구조를 트리 형태로 표현한 JSON과 이에 대한 여러 반론/질문들입니다. "
        "이러한 반론/질문들을 비판적 사고, 논리성, 관련성, 독창성 등을 기준으로 평가하여, "
        f"가장 적절하다고 생각되는 {review_num}개를 선택해 주세요.\n\n"
        f"[트리 구조]:\n{tree_str}\n\n"
        f"[반론/질문 목록]:\n{reviews_str}\n\n"
        f"Top {review_num}개의 반론/질문을 순위와 함께 JSON 배열 형태로 반환해주세요. 각 항목은 원래 반론/질문의 모든 정보를 포함해야 합니다."
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a critical thinking expert who can evaluate arguments and questions in Korean."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.3,  # Lower temperature for more consistent ranking
        max_completion_tokens=4096,
        top_p=1
    )
    
    try:
        ranked_reviews = json.loads(response.choices[0].message.content)
        # Ensure we have the expected format - should be a list of review objects
        if isinstance(ranked_reviews, list) and len(ranked_reviews) <= review_num:
            return ranked_reviews
        # If we get a different format (e.g., {"ranked_reviews": [...]}), try to extract the list
        elif isinstance(ranked_reviews, dict) and any(isinstance(ranked_reviews.get(k), list) for k in ranked_reviews):
            for k, v in ranked_reviews.items():
                if isinstance(v, list) and len(v) <= review_num:
                    return v
        
        # Fallback to original reviews if format is unexpected
        return reviews[:review_num]
    except Exception as e:
        print(f"Error parsing ranked reviews: {e}")
        return reviews[:review_num]

@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest):
    try:
        global previous_tree
        tree = request.tree
        review_num = request.review_num
        
        # Get current tree state as dictionary
        current_tree_dict = get_all_nodes(tree)
        
        # Determine which nodes to review
        if previous_tree:
            # Get previous tree state
            previous_tree_dict = get_all_nodes(previous_tree)
            
            # Find new nodes
            new_nodes = find_new_nodes(current_tree_dict, previous_tree_dict)
            
            if not new_nodes:
                # If no new nodes, review root level nodes as fallback
                new_nodes = [tree] + tree.child
        else:
            # First time seeing any tree, review the main nodes
            new_nodes = [tree] + tree.child
        
        # Generate reviews for new nodes in parallel
        review_tasks = [generate_review(node, tree) for node in new_nodes]
        reviews = await asyncio.gather(*review_tasks)
        
        # Flatten the list of reviews if any are lists themselves
        flattened_reviews = []
        for review in reviews:
            if isinstance(review, list):
                flattened_reviews.extend(review)
            else:
                flattened_reviews.append(review)
        
        # Rank the reviews and get the top ones
        ranked_reviews = await rank_reviews(flattened_reviews, tree, review_num)
        
        # Store the current tree for future comparison
        previous_tree = deepcopy(tree)
        
        # Return the ranked reviews
        return {"data": ranked_reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating review: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)