import json
import asyncio
from typing import List, Dict, Any
from config import client, OPENAI_MODEL
from models import TreeNode
from services.tree_utils import extract_subtree_to_root


async def generate_summary(content: str) -> str:
    """Generate a summary for a single content string."""
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Summarize the following content into a very concise single sentence in Korean."},
            {"role": "user", "content": content}
        ]
    )
    return response.choices[0].message.content


async def generate_summaries(contents: List[str]) -> List[str]:
    """Generate summaries for multiple content strings in parallel."""
    tasks = [generate_summary(content) for content in contents]
    return await asyncio.gather(*tasks)


async def generate_chat_response(messages: List[Dict[str, Any]]) -> str:
    """Generate a chat response using OpenAI."""
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages
    )
    return response.choices[0].message.content


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

    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
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
        temperature=0.7,
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
    
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
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
