import json
import asyncio
import re
from typing import List, Dict, Any
from config import client, perplexity_client, cerebras_client, OPENAI_MODEL, PERPLEXITY_MODEL, CEREBRAS_MODEL
from models import TreeNode
from services.tree_utils import extract_subtree_to_root


def clean_cerebras_response(response: str) -> str:
    """Clean the Cerebras API response by removing <think> tags and surrounding newlines."""
    # Remove <think>\n\n</think> and surrounding newlines
    cleaned = re.sub(r'<think>\n\n</think>\n\n', '', response)
    return cleaned


async def generate_cerebras_summary(content: str) -> str:
    """Generate a summary for a single content string using Cerebras API."""
    response = await cerebras_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Summarize the following content into a very concise single sentence in Korean. /nothink"
            },
            {
                "role": "user",
                "content": content
            },
        ],
        model=CEREBRAS_MODEL,
        stream=False,
        max_tokens=16382,
        temperature=0.6,
        top_p=0.95
    )
    
    # Extract and clean the response content
    raw_content = response.choices[0].message.content
    return clean_cerebras_response(raw_content)


async def generate_summary(content: str) -> str:
    """Generate a summary for a single content string."""
    return await generate_cerebras_summary(content)


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


async def generate_initial_rebuttal(node: TreeNode, tree: TreeNode, persona: str) -> str:
    """Generate an initial rebuttal with a specific persona as plain text (1-2 lines)."""
    # Extract the subtree from the node to the root
    subtree = extract_subtree_to_root(node, tree)
    subtree_json = subtree.model_dump()
    subtree_str = json.dumps(subtree_json, ensure_ascii=False, indent=2)
    
    persona_prompt = ""
    if persona == "teacher_rebuttal":
        persona_prompt = "당신은 전문 지식을 갖춘 선생님으로, 학생이 이해하기 쉽게 논리적인 반론을 제시합니다."
    elif persona == "student_rebuttal":
        persona_prompt = "당신은 비판적 사고력을 갖춘 학생으로, 다른 학생이 이해하기 쉽게 반론을 제시합니다."
    
    prompt = (
        f"{persona_prompt}\n\n"
        "다음은 논증 구조를 트리 형태로 표현한 JSON입니다. "
        "각 노드는 id, type, content, summary, child, sibling 등의 정보를 포함합니다. "
        "아래 트리 구조는 검토 대상 노드부터 루트까지의 경로를 포함하는 서브트리입니다. "
        "이를 참고하여, 주어진 nodeId(검토 대상 노드)에 대해 비판적 사고를 바탕으로 "
        "반론(반박)을 한국어로 생성하세요. 반드시 반박 형태로 작성해야 합니다.\n"
        "중요: 반론은 오직 주어진 nodeId에 있는 주장의 근거에 대해서만 작성해야 하며, 예상 반론의 근거에 대해서는 작성하지 마세요.\n"
        "응답은 반드시 1-2줄의 간결한 텍스트 형식으로 작성해주세요.\n"
        "학생이 이해하기 쉬운 언어로 작성해주세요.\n"
        f"[트리 구조]:\n{subtree_str}\n"
        f"[검토 대상 nodeId]: {node.id}"
    )
    
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a critical thinker who generates concise counterarguments in Korean. You always produce rebuttals, never questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_completion_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content.strip()


async def generate_initial_question(node: TreeNode, tree: TreeNode, persona: str) -> str:
    """Generate an initial question with a specific persona as plain text (1-2 lines)."""
    # Extract the subtree from the node to the root
    subtree = extract_subtree_to_root(node, tree)
    subtree_json = subtree.model_dump()
    subtree_str = json.dumps(subtree_json, ensure_ascii=False, indent=2)
    
    persona_prompt = ""
    if persona == "teacher_question":
        persona_prompt = "당신은 전문 지식을 갖춘 선생님으로, 학생이 이해하기 쉽게 깊이 있는 질문을 제시합니다."
    elif persona == "student_question":
        persona_prompt = "당신은 호기심 많은 학생으로, 다른 학생이 이해하기 쉽게 질문을 제시합니다."
    
    prompt = (
        f"{persona_prompt}\n\n"
        "다음은 논증 구조를 트리 형태로 표현한 JSON입니다. "
        "각 노드는 id, type, content, summary, child, sibling 등의 정보를 포함합니다. "
        "아래 트리 구조는 검토 대상 노드부터 루트까지의 경로를 포함하는 서브트리입니다. "
        "이를 참고하여, 주어진 nodeId(검토 대상 노드)에 대해 비판적 사고를 바탕으로 "
        "날카로운 질문을 한국어로 생성하세요. 반드시 질문 형태로 작성해야 합니다.\n"
        "중요: 질문은 오직 주어진 nodeId에 있는 주장의 근거에 대해서만 작성해야 하며, 예상 반론의 근거에 대해서는 작성하지 마세요.\n"
        "응답은 반드시 1-2줄의 간결한 텍스트 형식으로 작성해주세요.\n"
        "학생이 이해하기 쉬운 언어로 작성해주세요.\n"
        f"[트리 구조]:\n{subtree_str}\n"
        f"[검토 대상 nodeId]: {node.id}"
    )
    
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a critical thinker who generates insightful questions in Korean. You always produce questions, never rebuttals."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_completion_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content.strip()


async def get_perplexity_search_results(query: str) -> str:
    """Get search results for a query using Perplexity API."""
    response = await perplexity_client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who can search for information and provide concise, informative results in Korean."},
            {"role": "user", "content": f"다음 주제에 대해 검색하고 관련 정보를 요약해주세요: {query}"}
        ],
        response_format={"type": "text"},
        temperature=0.5,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    return response.choices[0].message.content.strip()


async def enhance_review_with_search_results(initial_review: str, search_results: str, node: TreeNode, tree: TreeNode, is_question: bool) -> Dict[str, Any]:
    """Enhance a review with search results and format it as a JSON object."""
    # Extract the subtree from the node to the root
    subtree = extract_subtree_to_root(node, tree)
    subtree_json = subtree.model_dump()
    subtree_str = json.dumps(subtree_json, ensure_ascii=False, indent=2)
    
    # Determine review type for JSON structure
    review_type = "질문" if is_question else "반론"
    
    prompt = (
        "다음은 논증 구조를 트리 형태로 표현한 JSON입니다. "
        "각 노드는 id, type, content, summary, child, sibling 등의 정보를 포함합니다. "
        "아래 트리 구조는 검토 대상 노드부터 루트까지의 경로를 포함하는 서브트리입니다.\n"
        f"[트리 구조]:\n{subtree_str}\n"
        f"[검토 대상 nodeId]: {node.id}\n\n"
        f"[초기 {review_type}]:\n{initial_review}\n\n"
        f"[검색 결과]:\n{search_results}\n\n"
        f"위 정보를 바탕으로, 초기 {review_type}을 검색 결과를 활용하여 보강하고, "
        f"JSON 형식으로 작성해주세요. 학생이 이해하기 쉬운 언어로 작성해주세요.\n\n"
        f"중요: 질문 타입일 경우 절대 반론 구조를 포함하지 마세요. 질문은 단순히 질문만 포함해야 합니다.\n"
        f"반론 타입일 경우 질문 구조를 포함하지 마세요. 각 타입에 맞는 정확한 구조만 사용하세요.\n\n"
        f"JSON 형식은 다음과 같습니다:\n"
        f"- 질문인 경우:\n"
        "```json\n"
        "{\n"
        "  \"tree\": {\n"
        "    \"type\": \"질문\",\n"
        "    \"content\": \"질문 내용\",\n"
        "    \"summary\": \"질문 요약\"\n"
        "  }\n"
        "}\n"
        "```\n"
        f"- 반론인 경우:\n"
        "```json\n"
        "{\n"
        "  \"tree\": {\n"
        "    \"type\": \"반론\",\n"
        "    \"content\": \"반론 내용\",\n"
        "    \"summary\": \"반론 요약\",\n"
        "    \"child\": [\n"
        "      {\n"
        "        \"type\": \"근거\",\n"
        "        \"content\": \"근거 내용\",\n"
        "        \"summary\": \"근거 요약\"\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}\n"
        "```\n"
    )
    
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a critical thinker who can enhance counterarguments and questions in Korean with search results."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    
    # Parse the response and add the node's ID as the parent field
    review_data = json.loads(response.choices[0].message.content)
    review_data["parent"] = node.id
    
    return review_data


async def generate_reviews_with_personas(node: TreeNode, tree: TreeNode) -> List[Dict[str, Any]]:
    """Generate reviews with different personas in parallel."""
    personas = ["teacher_rebuttal", "teacher_question", "student_rebuttal", "student_question"]
    
    # Generate initial reviews with different personas in parallel
    print("\n=== Generating initial reviews with different personas ===")
    
    # Create a list to store all tasks
    initial_review_tasks = []
    
    # Add tasks for each persona, using the appropriate function
    for persona in personas:
        if persona.endswith("rebuttal"):
            initial_review_tasks.append(generate_initial_rebuttal(node, tree, persona))
        elif persona.endswith("question"):
            initial_review_tasks.append(generate_initial_question(node, tree, persona))
    
    # Wait for all tasks to complete
    initial_reviews = await asyncio.gather(*initial_review_tasks)
    
    # Print initial reviews for each persona
    for i, persona in enumerate(personas):
        persona_name = persona.replace("_", " ").title()
        print(f"\n[{persona_name}] Initial review:\n{initial_reviews[i]}")
    
    # Use each initial review as a search query and get search results
    search_tasks = [get_perplexity_search_results(review) for review in initial_reviews]
    search_results = await asyncio.gather(*search_tasks)
    
    # Enhance each review with its search results
    print("\n=== Enhancing reviews with search results ===")
    is_question_values = [persona.endswith('question') for persona in personas]
    enhanced_review_tasks = [
        enhance_review_with_search_results(
            initial_reviews[i], 
            search_results[i], 
            node, 
            tree, 
            is_question_values[i]
        ) for i in range(len(personas))
    ]
    enhanced_reviews = await asyncio.gather(*enhanced_review_tasks)
    
    # Add persona information and print enhanced reviews
    for i, review in enumerate(enhanced_reviews):
        persona = personas[i]
        persona_name = persona.replace("_", " ").title()
        review["persona"] = persona
        
        review_type = review.get("tree", {}).get("type", "unknown")
        review_content = review.get("tree", {}).get("content", "No content")
        print(f"\n[{persona_name}] Enhanced {review_type}:\n{review_content[:200]}...")
    
    return enhanced_reviews


async def select_best_review_for_evidence(reviews: List[Dict[str, Any]], node: TreeNode, tree: TreeNode) -> Dict[str, Any]:
    """Select the best review for a given evidence using Cerebras API."""
    print("\n=== Selecting best review for evidence ===")
    reviews_str = json.dumps(reviews, ensure_ascii=False, indent=2)
    subtree = extract_subtree_to_root(node, tree)
    subtree_json = subtree.model_dump()
    subtree_str = json.dumps(subtree_json, ensure_ascii=False, indent=2)
    
    prompt = (
        "다음은 한 근거 노드에 대해 다양한 페르소나로 생성된 여러 개의 반론 또는 질문입니다.\n"
        "주어진 노드와 트리 구조를 바탕으로, 가장 적절한 하나의 반론 또는 질문을 선택해주세요.\n"
        "선택 기준:\n"
        "1. 주장의 근거에 집중: 오직 주장의 근거에 대해서만 반론/질문하고 예상 반론의 근거에 대해서는 다루지 않았는가?\n"
        "2. 논리적 타당성: 근거와 직접적으로 관련이 있고 논리적으로 타당한가?\n"
        "3. 교육적 가치: 학생의 비판적 사고를 자극하고 학습에 도움이 되는가?\n"
        "4. 명확성: 학생이 이해하기 쉽게 작성되었는가?\n"
        "5. 독창성: 새로운 관점이나 통찰을 제공하는가?\n\n"
        f"[트리 구조]:\n{subtree_str}\n"
        f"[검토 대상 nodeId]: {node.id}\n"
        f"[반론/질문 목록]:\n{reviews_str}\n"
        "가장 적절한 반론 또는 질문의 인덱스(0부터 시작)를 응답해주세요."
    )
    
    response = await cerebras_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an educational assistant helping to select the most appropriate review (counterargument or question) for a given evidence. /nothink"
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        model=CEREBRAS_MODEL,
        stream=False,
        max_tokens=16382,
        temperature=0.1,
        top_p=0.95
    )
    
    # Extract and clean the response content
    raw_content = clean_cerebras_response(response.choices[0].message.content)
    
    # Extract the index from the response
    try:
        # Try to find a number in the response
        index_match = re.search(r'\d+', raw_content)
        if index_match:
            selected_index = int(index_match.group())
            if 0 <= selected_index < len(reviews):
                selected_review = reviews[selected_index]
                persona = selected_review.get("persona", "unknown")
                persona_name = persona.replace("_", " ").title()
                review_type = selected_review.get("tree", {}).get("type", "unknown")
                review_content = selected_review.get("tree", {}).get("content", "No content")
                
                print(f"\n=== Selected Review for Evidence Node {node.id} ===")
                print(f"Selected Persona: {persona_name}")
                print(f"Review Type: {review_type}")
                print(f"Content: {review_content[:200]}...")
                return selected_review
        
        # Fallback to the first review if no valid index is found
        print("\nWarning: No valid index found, falling back to first review.")
        selected_review = reviews[0]
        persona = selected_review.get("persona", "unknown")
        persona_name = persona.replace("_", " ").title()
        print(f"Fallback Persona: {persona_name}")
        return selected_review
    except Exception as e:
        print(f"Error selecting best review: {e}")
        print("Falling back to first review due to error.")
        return reviews[0]


async def select_best_overall_review(selected_reviews: List[Dict[str, Any]], tree: TreeNode) -> Dict[str, Any]:
    """Select the best overall review from the selected reviews for each evidence."""
    print("\n=== Selecting best overall review ===")
    
    if len(selected_reviews) <= 1:
        if selected_reviews:
            selected_review = selected_reviews[0]
            persona = selected_review.get("persona", "unknown")
            persona_name = persona.replace("_", " ").title()
            review_type = selected_review.get("tree", {}).get("type", "unknown")
            print(f"\nOnly one review available - automatically selected:")
            print(f"Persona: {persona_name}")
            print(f"Review Type: {review_type}")
            return selected_review
        return None
    
    # Print all available reviews with their personas
    print("\nAvailable reviews for selection:")
    for i, review in enumerate(selected_reviews):
        persona = review.get("persona", "unknown")
        persona_name = persona.replace("_", " ").title()
        review_type = review.get("tree", {}).get("type", "unknown")
        parent_id = review.get("parent", "unknown")
        print(f"{i}. [{persona_name}] {review_type} for node {parent_id}")
    
    reviews_str = json.dumps(selected_reviews, ensure_ascii=False, indent=2)
    tree_json = tree.model_dump()
    tree_str = json.dumps(tree_json, ensure_ascii=False, indent=2)
    
    prompt = (
        "다음은 여러 근거 노드에 대해 선택된 반론 또는 질문입니다.\n"
        "주어진 트리 구조를 바탕으로, 전체 논증 구조에서 가장 적절한 하나의 반론 또는 질문을 선택해주세요.\n"
        "선택 기준:\n"
        "1. 주장의 근거에 집중: 오직 주장의 근거에 대해서만 반론/질문하고 예상 반론의 근거에 대해서는 다루지 않았는가?\n"
        "2. 논증 구조에서의 중요성: 논증 구조의 핵심 주장이나 근거에 관련된 반론/질문인가?\n"
        "3. 논리적 타당성: 논리적으로 타당하고 설득력이 있는가?\n"
        "4. 교육적 가치: 학생의 비판적 사고를 자극하고 학습에 도움이 되는가?\n"
        "5. 명확성: 학생이 이해하기 쉽게 작성되었는가?\n\n"
        f"[트리 구조]:\n{tree_str}\n"
        f"[반론/질문 목록]:\n{reviews_str}\n"
        "가장 적절한 반론 또는 질문의 인덱스(0부터 시작)를 응답해주세요."
    )
    
    response = await cerebras_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an educational assistant helping to select the most appropriate review (counterargument or question) from a list of reviews. /nothink"
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        model=CEREBRAS_MODEL,
        stream=False,
        max_tokens=16382,
        temperature=0.1,
        top_p=0.95
    )
    
    # Extract and clean the response content
    raw_content = clean_cerebras_response(response.choices[0].message.content)
    
    # Extract the index from the response
    try:
        # Try to find a number in the response
        index_match = re.search(r'\d+', raw_content)
        if index_match:
            selected_index = int(index_match.group())
            if 0 <= selected_index < len(selected_reviews):
                selected_review = selected_reviews[selected_index]
                persona = selected_review.get("persona", "unknown")
                persona_name = persona.replace("_", " ").title()
                review_type = selected_review.get("tree", {}).get("type", "unknown")
                content = selected_review.get("tree", {}).get("content", "No content")
                parent_id = selected_review.get("parent", "unknown")
                
                print(f"\n=== Best Overall Review Selected ===")
                print(f"Selected Index: {selected_index}")
                print(f"Persona: {persona_name}")
                print(f"Review Type: {review_type}")
                print(f"For Node: {parent_id}")
                print(f"Content Preview: {content[:200]}...")
                
                return selected_review
        
        # Fallback to the first review if no valid index is found
        print("\nWarning: No valid index found in Cerebras response, falling back to first review.")
        selected_review = selected_reviews[0]
        persona = selected_review.get("persona", "unknown")
        persona_name = persona.replace("_", " ").title()
        print(f"Fallback Persona: {persona_name}")
        return selected_review
    except Exception as e:
        print(f"Error selecting best overall review: {e}")
        print("Falling back to first review due to error.")
        return selected_reviews[0]


def validate_review_type(review: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fix review type to ensure it adheres to the correct structure."""
    persona = review.get("persona", "")
    tree_data = review.get("tree", {})
    review_type = tree_data.get("type", "")
    
    # For question types, ensure no child nodes
    if "question" in persona or review_type == "질문":
        # Force the type to be "질문"
        tree_data["type"] = "질문"
        
        # Remove any child nodes if present
        if "child" in tree_data:
            del tree_data["child"]
            print(f"Removed child nodes from question type for consistency")
    
    # For rebuttal types, ensure proper structure
    elif "rebuttal" in persona or review_type == "반론":
        # Force the type to be "반론"
        tree_data["type"] = "반론"
        
        # Ensure child array exists with at least one evidence node
        if "child" not in tree_data or not tree_data["child"]:
            tree_data["child"] = [{
                "type": "근거",
                "content": "이 반론에 대한 근거가 필요합니다.",
                "summary": "근거 필요"
            }]
            print(f"Added missing evidence node to rebuttal")
        
        # Ensure all child nodes are evidence type
        for child in tree_data["child"]:
            if child.get("type") != "근거":
                child["type"] = "근거"
                print(f"Fixed child node type to '근거'")
    
    # Update the tree in the review
    review["tree"] = tree_data
    return review


async def generate_review(node: TreeNode, tree: TreeNode) -> Dict[str, Any]:
    """Generate a review (counterargument or question) for a given node using the new workflow."""
    print(f"Generating review for node {node.id} with new workflow")
    
    # Step 1: Generate reviews with different personas in parallel
    reviews = await generate_reviews_with_personas(node, tree)
    print(f"Generated {len(reviews)} reviews with different personas")
    
    # Step 2: Select the best review for this evidence
    selected_review = await select_best_review_for_evidence(reviews, node, tree)
    print(f"Selected best review of type: {selected_review.get('tree', {}).get('type', 'unknown')}")
    
    # Step 3: Validate and fix the review type to ensure correct structure
    validated_review = validate_review_type(selected_review)
    
    return validated_review


async def rank_reviews(reviews: List[Dict[str, Any]], tree: TreeNode, review_num: int) -> List[Dict[str, Any]]:
    """Rank the generated reviews and return the top ones."""
    if len(reviews) == 0:
        return []
    
    if len(reviews) == 1:
        return reviews
    
    # If we have more than one review but fewer than or equal to the requested number
    if 1 < len(reviews) <= review_num:
        # Use select_best_overall_review to select the best review
        best_review = await select_best_overall_review(reviews, tree)
        # If we're supposed to return all of them anyway, put the best one first
        ordered_reviews = [best_review]
        for review in reviews:
            if review != best_review:
                ordered_reviews.append(review)
        return ordered_reviews
    
    # For cases where we have more reviews than requested number
    
    tree_json = tree.model_dump()
    tree_str = json.dumps(tree_json, ensure_ascii=False, indent=2)
    reviews_str = json.dumps(reviews, ensure_ascii=False, indent=2)
    
    system_content = [
        {
            "type": "text",
            "text": "주어진 논증 구조를 트리 형태로 표현한 JSON과 반론/질문 목록을 평가하여 가장 우수한 항목의 순위를 매기세요.\n\n주어진 반론/질문은 비판적 사고, 논리성, 관련성, 독창성 등을 기준으로 종합적으로 평가해야 합니다. 평가를 바탕으로 반론/질문들을 정렬하고, 상위 n개의 순위를 도출하세요.\n\n# Steps\n\n1. **Input Parsing**: 수신된 JSON 형식의 논증 구조와 반론/질문 목록을 파싱합니다.\n2. **Critique and Evaluate**:\n   - 각 반론/질문이 오직 주장의 근거에 대해서만 작성되었고 예상 반론의 근거에 대해서는 다루지 않았는지 확인합니다.\n   - 각 반론/질문에 대해 비판적 사고 적용 정도를 분석합니다.\n   - 논리적 구조와 일관성을 평가합니다.\n   - 논증과의 직접적 관련성을 검사합니다.\n   - 독창성과 새로운 관점을 평가합니다.\n3. **Scoring**: 위의 기준들을 종합하여 각 반론/질문의 점수를 산출합니다. 주장의 근거에만 집중하고 예상 반론의 근거는 다루지 않은 반론/질문에 가장 높은 점수를 부여합니다.\n4. **Ranking**: 점수에 따라 반론/질문을 정렬하고, 상위 n 개의 항목을 리스트업합니다.\n\n# Output Format\n\n출력은 상위 review_num개의 반론/질문에 대한 순위로, 각 항목에 대해 해당 순위를 숫자로 배열 형태로 제공합니다. \n\n예를 들어:\n```json\n{\n  \"ranked_reviews\": [3, 2, 5]\n}\n```\n여기서 숫자는 반론/질문의 인덱스를 나타냅니다.\n\n# Notes\n\n- 평가 기준을 균형 있게 고려하여 점수를 매기십시오.\n- 반론/질문이 예상 반론의 근거를 다루거나, 논증과 직접적으로 관련이 없거나 독창성이 부족할 경우, 낮은 점수를 부여하십시오.\n- 입력 데이터는 유효한 포맷이어야 하며, 오류 검사를 포함하십시오."
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
