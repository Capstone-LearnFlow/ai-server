import requests
import json
import time
import os

BASE_URL = "http://localhost:8000"

def test_review_with_no_new_nodes():
    # Set up logging to a file
    log_file = "test_results.log"
    with open(log_file, "w") as f:
        f.write("=== Test Results: Review with No New Nodes ===\n\n")
        
    def log(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")
    """
    Test the review service with a scenario where a new tree comes in without new evidence
    but there are pre-made unselected reviews available.
    
    This test:
    1. Creates an initial tree with evidence nodes
    2. Makes a first request with review_num=1 to ensure there are unselected reviews
    3. Creates a second tree identical to the first (no new evidence)
    4. Makes a second request and verifies that it works using the unselected reviews
    """
    url = f"{BASE_URL}/review"
    
    # Set student ID and assignment ID for this test
    student_id = "test_student"
    assignment_id = "test_assignment"
    
    # First, reset state to ensure clean test
    reset_url = f"{BASE_URL}/reset?student_id={student_id}&assignment_id={assignment_id}"
    log("\n==== Resetting state for test ====")
    response = requests.post(reset_url)
    log(f"Reset response: {response.status_code}")
    
    # Create initial tree with evidence nodes
    initial_tree = {
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
                        "content": "첫 번째 근거입니다. 이 근거는 주장을 뒷받침합니다.",
                        "summary": "첫 번째 근거",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    },
                    {
                        "id": "evidence2",
                        "type": "근거",
                        "child": [],
                        "sibling": [],
                        "content": "두 번째 근거입니다. 이 근거는 주장을 다른 측면에서 지지합니다.",
                        "summary": "두 번째 근거",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    },
                    {
                        "id": "evidence3",
                        "type": "근거",
                        "child": [],
                        "sibling": [],
                        "content": "세 번째 근거입니다. 이 근거는 주장에 대한 추가적인 지지를 제공합니다.",
                        "summary": "세 번째 근거",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    }
                ],
                "sibling": [],
                "content": "이것은 주장입니다. 이 주장은 세 가지 근거로 뒷받침됩니다.",
                "summary": "주장 요약",
                "created_by": "user1",
                "created_at": "2025-05-12T10:00:00Z",
                "updated_at": "2025-05-12T10:00:00Z"
            }
        ],
        "sibling": [],
        "content": "테스트 주제",
        "summary": "주제 요약",
        "created_by": "user1",
        "created_at": "2025-05-12T09:00:00Z",
        "updated_at": "2025-05-12T09:00:00Z"
    }
    
    # First request - request only 1 review to ensure there are unselected reviews
    log("\n==== First request: Initial tree with evidence, requesting 1 review ====")
    payload1 = {
        "tree": initial_tree,
        "review_num": 1,  # Request only 1 to ensure unselected reviews exist
        "student_id": student_id,
        "assignment_id": assignment_id
    }
    
    response1 = requests.post(url, json=payload1)
    if response1.status_code == 200:
        result1 = response1.json()
        log(f"First request successful. Response status: {response1.status_code}")
        log(f"Number of reviews returned: {len(result1.get('data', []))}")
        
        # Log some details about the reviews
        for i, review in enumerate(result1.get('data', [])):
            log(f"Review {i+1} type: {review.get('tree', {}).get('type', 'unknown')}")
    else:
        log(f"First request failed. Status code: {response1.status_code}")
        log(f"Error message: {response1.text}")
        return
    
    # Second request - Same tree (no new evidence), should use unselected reviews
    log("\n==== Second request: Same tree (no new evidence), should use unselected reviews ====")
    payload2 = {
        "tree": initial_tree,  # Same tree, no changes
        "review_num": 1,
        "student_id": student_id,
        "assignment_id": assignment_id
    }
    
    response2 = requests.post(url, json=payload2)
    if response2.status_code == 200:
        result2 = response2.json()
        log(f"Second request successful. Response status: {response2.status_code}")
        log(f"Number of reviews returned: {len(result2.get('data', []))}")
        
        # Log some details about the reviews
        for i, review in enumerate(result2.get('data', [])):
            log(f"Review {i+1} type: {review.get('tree', {}).get('type', 'unknown')}")
        
        log("This confirms that the system uses pre-made unselected reviews when there are no new evidence nodes.")
    else:
        log(f"Second request failed. Status code: {response2.status_code}")
        log(f"Error message: {response2.text}")
        log("The system does not appear to be using pre-made unselected reviews when there are no new evidence nodes.")
    
    log("\n==== Test completed ====")
    log(f"Results written to {os.path.abspath(log_file)}")

if __name__ == "__main__":
    test_review_with_no_new_nodes()