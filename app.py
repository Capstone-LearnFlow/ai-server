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

# In-memory storage for previous tree and unselected reviews
previous_tree = None
unselected_reviews = []

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

def extract_subtree_to_root(node: TreeNode, tree: TreeNode) -> TreeNode:
    """Extract a subtree that includes the path from the given node to the root.
    
    Args:
        node: The node to start from
        tree: The complete tree structure
        
    Returns:
        A new tree containing only the path from the node to the root
    """
    # Create a parent map for the entire tree
    parent_map = get_parent_map(tree)
    
    # Build a path from the node to the root
    path_nodes = [node]
    current_id = node.id
    
    while current_id in parent_map and parent_map[current_id]:
        parent_node = parent_map[current_id]
        path_nodes.append(parent_node)
        current_id = parent_node.id
    
    # Create a new subtree with only the nodes in the path
    # Start from the root (last node in the path)
    if not path_nodes:
        return deepcopy(node)  # Fallback to just the node if path is empty
    
    path_nodes.reverse()  # Reverse so root is first
    
    # Create a copy of the root node
    subtree = deepcopy(path_nodes[0])
    subtree.child = []  # Clear children
    subtree.sibling = []  # Clear siblings
    
    # Build the path down to the target node
    current_node = subtree
    for i in range(1, len(path_nodes)):
        child_copy = deepcopy(path_nodes[i])
        child_copy.child = []  # Clear children that aren't in the path
        child_copy.sibling = []  # Clear siblings
        
        current_node.child = [child_copy]  # Add as the only child
        current_node = child_copy  # Move down to the child for next iteration
    
    # If the last node in the path is not the original node, add it
    if path_nodes[-1].id != node.id:
        node_copy = deepcopy(node)
        node_copy.child = []
        node_copy.sibling = []
        current_node.child = [node_copy]
    
    return subtree

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

def get_parent_map(tree: TreeNode) -> Dict[str, TreeNode]:
    """Build a dictionary mapping node IDs to their parent nodes."""
    parent_map = {}
    
    def process_children(parent_node, children):
        for child in children:
            parent_map[child.id] = parent_node
            # Process this child's children
            process_children(child, child.child)
            # Process this child's siblings
            process_children(parent_node, child.sibling)
    
    # Process root node's children
    process_children(tree, tree.child)
    
    # Process root node's siblings
    for sibling in tree.sibling:
        parent_map[sibling.id] = None  # Root-level siblings don't have a parent in our context
        process_children(sibling, sibling.child)
    
    return parent_map

def find_new_nodes(current_tree: Dict[str, TreeNode], previous_tree: Dict[str, TreeNode]) -> List[TreeNode]:
    """Find nodes that were added since the previous tree state."""
    new_nodes = []
    
    # Build a dictionary of node IDs to their parent nodes
    # We need to reconstruct the tree first
    tree_root = None
    for node_id, node in current_tree.items():
        if tree_root is None or len(node.sibling) > 0:  # Simple heuristic to find a root node
            tree_root = node
            break
    
    if tree_root is None and current_tree:  # If we couldn't find a root but have nodes
        tree_root = next(iter(current_tree.values()))  # Just take the first node
        
    parent_map = get_parent_map(tree_root) if tree_root else {}
    
    for node_id, node in current_tree.items():
        # Check if this is a '근거' node with a '반론' parent, which we want to exclude
        should_exclude = False
        if node.type == "근거" and node_id in parent_map:
            parent_node = parent_map[node_id]
            if parent_node and parent_node.type == "반론":
                should_exclude = True
                
        # If node didn't exist before or is of type 근거 or 답변 (and not excluded) and has been updated
        if not should_exclude and (
            node_id not in previous_tree or 
            (node.type in ["근거", "답변"] and 
             (node.updated_at != previous_tree[node_id].updated_at if node_id in previous_tree else True))
        ):
            new_nodes.append(node)

    return new_nodes

async def generate_review(node: TreeNode, tree: TreeNode) -> Dict[str, Any]:
    """Generate a review (counterargument or question) for a given node."""
    # Extract the subtree from the node to the root
    subtree = extract_subtree_to_root(node, tree)
    subtree_json = subtree.model_dump()
    subtree_str = json.dumps(subtree_json, ensure_ascii=False, indent=2)
    
    prompt = (
        "다음은 논증 구조를 트리 형태로 표현한 JSON입니다. "
        "각 노드는 id, type, content, summary, child, sibling 등의 정보를 포함합니다. "
        "아래 트리 구조는 검토 대상 노드부터 루트까지의 경로를 포함하는 서브트리입니다. "
        "이를 참고하여, 주어진 nodeId(검토 대상 노드)에 대해 비판적 사고를 바탕으로 반론(반박) 또는 날카로운 질문을 한국어로 생성하세요. "
        "IMPORTANT: 반론에는 반드시 child에 type이 근거인 노드가 포함되어야 하며, 질문은 명확하고 구체적이어야 합니다.\n"
        "응답은 반드시 JSON 스키마에 맞춰주세요.\n"
        f"[트리 구조]:\n{subtree_str}\n"
        f"[검토 대상 nodeId]: {node.id}"
    )
    print("Prompt for review generation:", prompt)

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
                    "required": ["tree"],
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
    
    # Parse the response and add the node's ID as the parent field
    review_data = json.loads(response.choices[0].message.content)
    review_data["parent"] = node.id
    
    return review_data

async def rank_reviews(reviews: List[Dict[str, Any]], tree: TreeNode, review_num: int) -> List[Dict[str, Any]]:
    """Rank the generated reviews and return the top ones."""
    if len(reviews) <= review_num:
        return reviews
    
    tree_json = tree.model_dump()
    tree_str = json.dumps(tree_json, ensure_ascii=False, indent=2)
    reviews_str = json.dumps(reviews, ensure_ascii=False, indent=2)
    
    system_content = [
        {
            "type": "text",
            "text": "주어진 논증 구조를 트리 형태로 표현한 JSON과 반론/질문 목록을 평가하여 가장 우수한 항목의 순위를 매기세요.\n\n주어진 반론/질문은 비판적 사고, 논리성, 관련성, 독창성 등을 기준으로 종합적으로 평가해야 합니다. 평가를 바탕으로 반론/질문들을 정렬하고, 상위 n개의 순위를 도출하세요.\n\n# Steps\n\n1. **Input Parsing**: 수신된 JSON 형식의 논증 구조와 반론/질문 목록을 파싱합니다.\n2. **Critique and Evaluate**:\n   - 각 반론/질문에 대해 비판적 사고 적용 정도를 분석합니다.\n   - 논리적 구조와 일관성을 평가합니다.\n   - 논증과의 직접적 관련성을 검사합니다.\n   - 독창성과 새로운 관점을 평가합니다.\n3. **Scoring**: 위의 기준들을 종합하여 각 반론/질문의 점수를 산출합니다.\n4. **Ranking**: 점수에 따라 반론/질문을 정렬하고, 상위 n 개의 항목을 리스트업합니다.\n\n# Output Format\n\n출력은 상위 review_num개의 반론/질문에 대한 순위로, 각 항목에 대해 해당 순위를 숫자로 배열 형태로 제공합니다. \n\n예를 들어:\n```json\n{\n  \"ranked_reviews\": [3, 2, 5]\n}\n```\n여기서 숫자는 반론/질문의 인덱스를 나타냅니다.\n\n# Notes\n\n- 평가 기준을 균형 있게 고려하여 점수를 매기십시오.\n- 반론/질문이 논증과 직접적으로 관련이 없거나 독창성이 부족할 경우, 낮은 점수를 부여하십시오.\n- 입력 데이터는 유효한 포맷이어야 하며, 오류 검사를 포함하십시오."
        }
    ]
    
    user_content = f"[트리 구조]:\n{tree_str}\n\n[반론/질문 목록]:\n{reviews_str}\n\n상위 {review_num}개의 반론/질문에 대한 순위를 도출해주세요."
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ranked_reviews_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "ranked_reviews": {
                            "type": "array",
                            "description": "An array of ranked reviews, each represented as a number.",
                            "items": {
                                "type": "number",
                                "description": "A single ranked review score."
                            }
                        }
                    },
                    "required": [
                        "ranked_reviews"
                    ],
                    "additionalProperties": False
                }
            }
        },
        temperature=0,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    try:
        parsed_response = json.loads(response.choices[0].message.content)
        ranked_indices = parsed_response.get("ranked_reviews", [])
        
        if not ranked_indices or len(ranked_indices) == 0:
            print("Warning: No ranked indices returned, falling back to first reviews")
            return reviews[:review_num]
        
        # Get the top ranked reviews based on the indices
        # Ensure indices are valid (within range and integer)
        valid_indices = [int(idx) for idx in ranked_indices if 0 <= int(idx) < len(reviews)][:review_num]
        if not valid_indices:
            return reviews[:review_num]
        
        # Get the selected reviews
        selected_reviews = [reviews[idx] for idx in valid_indices if idx < len(reviews)]
        
        # If we have fewer selected reviews than requested, add from the beginning of the list
        if len(selected_reviews) < review_num:
            remaining_indices = [i for i in range(len(reviews)) if i not in valid_indices]
            for i in remaining_indices:
                if len(selected_reviews) >= review_num:
                    break
                selected_reviews.append(reviews[i])
        
        return selected_reviews
        
    except Exception as e:
        print(f"Error parsing ranked reviews: {e}")
        return reviews[:review_num]

@app.post("/review", response_model=ReviewResponse)
async def review(request: ReviewRequest):
    try:
        global previous_tree, unselected_reviews
        tree = request.tree
        review_num = request.review_num
        
        # Get current tree state as dictionary
        current_tree_dict = get_all_nodes(tree)
        print("previous_tree:", previous_tree)
        
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
            # First time seeing any tree, filter for nodes of type '근거' or '답변'
            new_nodes = []
            for node_id, node in current_tree_dict.items():
                if node.type in ["근거", "답변"]:
                    new_nodes.append(node)
            
            # If no 근거 or 답변 nodes are found, fall back to main nodes
            #if not new_nodes:
            #    new_nodes = [tree] + tree.child
        print("new_nodes:", new_nodes)
        
        # Generate reviews for new nodes in parallel
        review_tasks = [generate_review(node, tree) for node in new_nodes]
        reviews = await asyncio.gather(*review_tasks)
        print("reviews:", reviews)
        
        # Flatten the list of reviews if any are lists themselves
        flattened_reviews = []
        for review in reviews:
            if isinstance(review, list):
                flattened_reviews.extend(review)
            else:
                flattened_reviews.append(review)
        
        # Combine newly generated reviews with previously unselected reviews
        combined_reviews = flattened_reviews + unselected_reviews
        print(f"Combined reviews (new + unselected): {len(combined_reviews)}")
        
        # Rank the reviews and get the top ones
        ranked_reviews = await rank_reviews(combined_reviews, tree, review_num)
        print("ranked_reviews:", ranked_reviews)
        
        # Store unselected reviews for future use
        if len(combined_reviews) > len(ranked_reviews):
            # Find reviews that weren't selected
            unselected_reviews = []
            selected_ids = {review.get("parent", "") for review in ranked_reviews}
            
            for review in combined_reviews:
                review_id = review.get("parent", "")
                review_content = review.get("tree", {}).get("content", "")
                
                # Check if this review isn't in the selected list
                # or if it's a different review for the same parent node
                is_duplicate = False
                for selected_review in ranked_reviews:
                    if (selected_review.get("parent", "") == review_id and 
                        selected_review.get("tree", {}).get("content", "") == review_content):
                        is_duplicate = True
                        break
                
                if not is_duplicate and review not in ranked_reviews:
                    unselected_reviews.append(review)
            
            print(f"Stored {len(unselected_reviews)} unselected reviews for future use")
        
        # Store the current tree for future comparison
        previous_tree = deepcopy(tree)
        
        # Return the ranked reviews
        return {"data": ranked_reviews}
    except Exception as e:
        print(f"Error in review endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating review: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)