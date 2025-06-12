import requests
import json
import os
import re
import time

BASE_URL = "http://localhost:8000"

def test_review_focus_on_main_evidence():
    """
    Test to verify that reviews/questions focus only on the main claim's evidence
    rather than anticipated counterarguments.
    
    This test:
    1. Creates a tree with a main claim and supporting evidence
    2. Requests reviews/questions for the evidence
    3. Analyzes the responses to verify they focus on the main claim's evidence, not anticipated counterarguments
    """
    # Set up logging to a file
    log_file = "test_results.log"
    with open(log_file, "w") as f:
        f.write("=== Test Results: Review Focus on Main Claim Evidence ===\n\n")
    
    def log(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")
    
    url = f"{BASE_URL}/review"
    
    # Set student ID and assignment ID for this test
    student_id = "test_student_focus"
    assignment_id = "test_assignment_focus"
    
    # First, reset state to ensure clean test
    reset_url = f"{BASE_URL}/reset?student_id={student_id}&assignment_id={assignment_id}"
    log("\n==== Resetting state for test ====")
    response = requests.post(reset_url)
    log(f"Reset response: {response.status_code}")
    
    # Create tree with a main claim and supporting evidence
    # The evidence will be what we want reviews/questions to focus on
    tree = {
        "id": "root",
        "type": "주제",
        "child": [
            {
                "id": "claim1",
                "type": "주장",
                "child": [
                    {
                        "id": "evidence1",
                        "type": "근거",
                        "child": [],
                        "sibling": [],
                        "content": "유전자 변형 작물은 농약 사용을 크게 줄이며, 이는 토양 오염을 줄이고 생태계를 보호하는 데 도움이 됩니다.",
                        "summary": "농약 사용 감소",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    },
                    {
                        "id": "evidence2",
                        "type": "근거",
                        "child": [],
                        "sibling": [],
                        "content": "유전자 변형 작물은 가뭄, 염분, 병충해에 더 강한 내성을 가지고 있어 기후 변화 조건에서도 생산성을 유지할 수 있습니다.",
                        "summary": "환경 스트레스 내성",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    }
                ],
                "sibling": [],
                "content": "유전자 변형 농산물(GMO)은 지속 가능한 농업을 위한 중요한 기술입니다.",
                "summary": "GMO의 중요성",
                "created_by": "user1",
                "created_at": "2025-05-12T10:00:00Z",
                "updated_at": "2025-05-12T10:00:00Z"
            }
        ],
        "sibling": [],
        "content": "유전자 변형 농산물(GMO)에 관한 논의",
        "summary": "GMO 논의",
        "created_by": "user1",
        "created_at": "2025-05-12T09:00:00Z",
        "updated_at": "2025-05-12T09:00:00Z"
    }
    
    # Request reviews for the evidence nodes
    # We'll request more reviews to get a mix of questions and counterarguments
    log("\n==== Requesting reviews ====")
    payload = {
        "tree": tree,
        "review_num": 2,  # Request 2 reviews to get a mix
        "student_id": student_id,
        "assignment_id": assignment_id
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        reviews = result.get("data", [])
        log(f"Request successful. Response status: {response.status_code}")
        log(f"Number of reviews returned: {len(reviews)}")
        
        # Analyze each review to verify it focuses on the main claim's evidence
        log("\n==== Analyzing reviews ====")
        
        for i, review in enumerate(reviews):
            review_type = review.get("tree", {}).get("type", "unknown")
            content = review.get("tree", {}).get("content", "")
            
            log(f"\nReview {i+1} - Type: {review_type}")
            log(f"Content: {content}")
            
            # Check if the review contains keywords that indicate focus on expected counterarguments
            # rather than the main claim's evidence
            counterargument_indicators = [
                "예상 반론", "예상되는 반론", "반대 의견", "예상 질문", "예상되는 질문",
                "반대 측", "반대편", "반대 입장", "반대 관점"
            ]
            
            found_indicators = []
            for indicator in counterargument_indicators:
                if indicator in content:
                    found_indicators.append(indicator)
            
            if found_indicators:
                log(f"WARNING: Review may focus on anticipated counterarguments. Indicators found: {', '.join(found_indicators)}")
                log("This suggests the review may not be focusing only on the main claim's evidence.")
            else:
                log("PASS: No indicators of focus on anticipated counterarguments found.")
                
            # Check if the review directly references the evidence content
            evidence_content = [
                tree["child"][0]["child"][0]["content"],
                tree["child"][0]["child"][1]["content"]
            ]
            
            evidence_references = []
            for e_content in evidence_content:
                # Create a simplified version for fuzzy matching
                simple_evidence = re.sub(r'[^\w\s]', '', e_content.lower())
                simple_content = re.sub(r'[^\w\s]', '', content.lower())
                
                # Check for significant word overlap
                evidence_words = set(simple_evidence.split())
                content_words = set(simple_content.split())
                overlap_words = evidence_words.intersection(content_words)
                
                if len(overlap_words) >= 3:  # At least 3 words in common
                    evidence_references.append(True)
                    break
            
            if evidence_references:
                log("PASS: Review directly references evidence content.")
            else:
                log("WARNING: Review may not directly reference evidence content.")
            
            log("-" * 50)
        
        # Overall assessment
        log("\n==== Overall Assessment ====")
        all_pass = all(
            "WARNING" not in log_line for review_idx in range(len(reviews)) 
            for log_line in open(log_file).readlines() 
            if f"Review {review_idx+1}" in log_line
        )
        
        if all_pass:
            log("SUCCESS: All reviews focus properly on the main claim's evidence, not on anticipated counterarguments.")
        else:
            log("PARTIAL SUCCESS or FAILURE: Some reviews may not be focusing correctly on the main claim's evidence.")
    else:
        log(f"Request failed. Status code: {response.status_code}")
        log(f"Error message: {response.text}")
    
    log("\n==== Test completed ====")
    log(f"Results written to {os.path.abspath(log_file)}")
    return os.path.abspath(log_file)

if __name__ == "__main__":
    test_review_focus_on_main_evidence()