import requests
import json
import os
import time

BASE_URL = "http://localhost:8000"

def test_exclude_nodes_with_counterarguments():
    """
    Test to verify that evidence nodes with existing counterarguments are excluded from review,
    even if they are new nodes.
    
    This test:
    1. Creates a tree with evidence nodes, some of which already have counterarguments
    2. Makes a review request
    3. Verifies that only evidence nodes without counterarguments are included in the review
    """
    # Set up logging to a file
    log_file = "test_results.log"
    with open(log_file, "w") as f:
        f.write("=== Test Results: Exclude Nodes with Counterarguments ===\n\n")
    
    def log(message):
        print(message)
        with open(log_file, "a") as f:
            f.write(message + "\n")
    
    url = f"{BASE_URL}/review"
    
    # Set student ID and assignment ID for this test
    student_id = "test_student_exclude"
    assignment_id = "test_assignment_exclude"
    
    # First, reset state to ensure clean test
    reset_url = f"{BASE_URL}/reset?student_id={student_id}&assignment_id={assignment_id}"
    log("\n==== Resetting state for test ====")
    response = requests.post(reset_url)
    log(f"Reset response: {response.status_code}")
    
    # Create a tree with evidence nodes, some with counterarguments
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
                        "child": [],  # No counterarguments
                        "sibling": [],
                        "content": "기후 변화 대응을 위한 재생 에너지 전환은 장기적 경제 효율성을 증가시킵니다.",
                        "summary": "경제 효율성 증가",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    },
                    {
                        "id": "evidence2",
                        "type": "근거",
                        "child": [
                            {
                                "id": "rebuttal1",
                                "type": "반론",  # This evidence has a counterargument
                                "child": [],
                                "sibling": [],
                                "content": "일부 재생 에너지 기술은 아직 비용 효율적이지 않으며, 지속적인 보조금 없이는 경쟁력이 부족합니다.",
                                "summary": "비용 효율성 부족",
                                "created_by": "user2",
                                "created_at": "2025-05-12T12:30:00Z",
                                "updated_at": "2025-05-12T12:30:00Z"
                            }
                        ],
                        "sibling": [],
                        "content": "재생 에너지로의 전환은 화석 연료 의존도를 줄여 에너지 가격 변동성을 감소시킵니다.",
                        "summary": "가격 변동성 감소",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    },
                    {
                        "id": "evidence3",
                        "type": "근거",
                        "child": [],  # No counterarguments
                        "sibling": [],
                        "content": "재생 에너지 전환은 대기 오염 감소와 건강 비용 절감으로 이어져 사회적 효용을 증가시킵니다.",
                        "summary": "사회적 효용 증가",
                        "created_by": "user1",
                        "created_at": "2025-05-12T11:30:00Z",
                        "updated_at": "2025-05-12T11:30:00Z"
                    }
                ],
                "sibling": [],
                "content": "재생 에너지로의 전환은 경제적으로 합리적인 전략입니다.",
                "summary": "경제적 합리성",
                "created_by": "user1",
                "created_at": "2025-05-12T10:00:00Z",
                "updated_at": "2025-05-12T10:00:00Z"
            }
        ],
        "sibling": [],
        "content": "재생 에너지 전환에 관한 논의",
        "summary": "재생 에너지 전환",
        "created_by": "user1",
        "created_at": "2025-05-12T09:00:00Z",
        "updated_at": "2025-05-12T09:00:00Z"
    }
    
    # Make a review request
    log("\n==== Making review request ====")
    payload = {
        "tree": tree,
        "review_num": 2,  # Request 2 reviews
        "student_id": student_id,
        "assignment_id": assignment_id
    }
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        reviews = result.get("data", [])
        log(f"Request successful. Response status: {response.status_code}")
        log(f"Number of reviews returned: {len(reviews)}")
        
        # Collect the target nodes of the reviews
        reviewed_nodes = [review.get("parent", "") for review in reviews]
        log(f"Nodes reviewed: {reviewed_nodes}")
        
        # Verify that evidence2 (which has a counterargument) is not included
        if "evidence2" in reviewed_nodes:
            log("FAIL: evidence2 was included in reviews despite having an existing counterargument")
        else:
            log("PASS: evidence2 was correctly excluded from reviews as it has an existing counterargument")
        
        # Verify that evidence1 and evidence3 (which don't have counterarguments) are included
        if "evidence1" in reviewed_nodes or "evidence3" in reviewed_nodes:
            log("PASS: At least one evidence without counterarguments was included in reviews")
        else:
            log("FAIL: No evidence without counterarguments was included in reviews")
        
        # Overall assessment
        log("\n==== Overall Assessment ====")
        if "evidence2" not in reviewed_nodes and ("evidence1" in reviewed_nodes or "evidence3" in reviewed_nodes):
            log("SUCCESS: The review service correctly excludes evidence nodes that already have counterarguments")
        else:
            log("FAILURE: The review service does not correctly handle evidence nodes with existing counterarguments")
    else:
        log(f"Request failed. Status code: {response.status_code}")
        log(f"Error message: {response.text}")
    
    log("\n==== Test completed ====")
    log(f"Results written to {os.path.abspath(log_file)}")
    return os.path.abspath(log_file)

if __name__ == "__main__":
    test_exclude_nodes_with_counterarguments()