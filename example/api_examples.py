import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Example for /chat endpoint
def example_chat():
    url = f"{BASE_URL}/chat"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}
        ],
    }
    response = requests.post(url, json=payload)
    print("/chat response:", response.json())

# Example for /summary endpoint
def example_summary():
    url = f"{BASE_URL}/summary"
    payload = {
        "contents": [
            "Traders will get their first sense of the initial inflationary effects of the tariffs with the release of key inflation data this week. April's Consumer Price Index (CPI) report is due Tuesday, followed by retail sales and the Producer Price Index (PPI) on Thursday.",
            "Brent (BZ=F) traded around $64 a barrel, paring an advance in early Asian trading after last week notching up its first weekly gain in three, while West Texas Intermediate (CL=F) was near $61. After negotiations in Geneva, US Treasury Secretary Scott Bessent and Trade Representative Jamieson Greer said that they were upbeat on progress and would share more information on Monday, with the positive sentiment echoed by their Chinese counterparts."
        ]
    }
    response = requests.post(url, json=payload)
    print("/summary response:", response.json())

# Example for /review endpoint
def example_review():
    url = f"{BASE_URL}/review"
    
    # 학생 ID와 과제 ID 설정
    # Set student ID and assignment ID
    student_id = "student001"
    assignment_id = "assignment001"
    
    # 초기 논증 트리 생성 - 주제와 주요 주장
    # Initial argument tree - topic and main argument
    initial_tree = {
        "id": "root",
        "type": "주제",
        "child": [
            {
                "id": "claim1",
                "type": "주장",
                "child": [],
                "sibling": [],
                "content": "정년 연장은 고령화 사회에서 노동력 확보와 노인 빈곤 문제 해결에 도움이 된다.",
                "summary": "정년 연장의 필요성",
                "created_by": "user1",
                "created_at": "2025-05-12T10:00:00Z",
                "updated_at": "2025-05-12T10:00:00Z"
            }
        ],
        "sibling": [],
        "content": "정년 연장에 관한 논의",
        "summary": "정년 연장 논의",
        "created_by": "user1",
        "created_at": "2025-05-12T09:00:00Z",
        "updated_at": "2025-05-12T09:00:00Z"
    }
    tree_with_evidence = json.loads(json.dumps(initial_tree))  # Deep copy
    tree_with_evidence["child"][0]["child"] = [
        {
            "id": "evidence1",
            "type": "근거",
            "child": [],
            "sibling": [],
            "content": "통계청 자료에 따르면 한국은 2025년부터 생산가능인구(15~64세)가 급격히 감소하여 2060년에는 현재의 약 65% 수준으로 줄어들 전망이다. 정년 연장은 이러한 인구구조 변화에 대응하는 필수적인 정책이다.",
            "summary": "생산가능인구 감소 통계",
            "created_by": "user1",
            "created_at": "2025-05-12T11:30:00Z",
            "updated_at": "2025-05-12T11:30:00Z"
        }
    ]
    tree_with_evidence["child"][0]["child"].append(
        {
            "id": "evidence2",
            "type": "근거",
            "child": [],
            "sibling": [],
            "content": "정년 연장은 고령화로 인한 노동력 부족 문제를 완화하는 데 효과적입니다. 2026년에는 65세 이상 인구가 전체의 20%를 넘을 것으로 예상되며, 정년 연장은 숙련된 인력을 계속 활용할 수 있는 방안으로 주목받고 있습니다",
            "summary": "노동력 부족 문제 완화",
            "created_by": "user1",
            "created_at": "2025-05-12T11:30:00Z",
            "updated_at": "2025-05-12T11:30:00Z"
        }
    )
    # 첫 번째 요청 - 기본 1개 리뷰
    # First request - default 1 review
    print("\n==== 첫 번째 요청: 초기 트리, 기본 1개 리뷰 ====")
    payload1 = {
        "tree": tree_with_evidence,
        "student_id": student_id,
        "assignment_id": assignment_id
    }
    response1 = requests.post(url, json=payload1)
    result1 = response1.json()
    print(json.dumps(result1, ensure_ascii=False, indent=2))
    
    # 트리에 새로운 근거 노드 추가
    # Add new evidence node to the tree
    tree_with_evidence = json.loads(json.dumps(tree_with_evidence))  # Deep copy
    tree_with_evidence["child"][0]["child"].append(
        {
            "id": "evidence3",
            "type": "근거",
            "child": [],
            "sibling": [],
            "content": "고령 근로자가 더 오래 일할 수 있도록 정년을 연장하면, 국민연금 수급 전까지의 소득 공백을 줄여 노인 빈곤 문제 해결에 도움이 됩니다. 이는 노후 소득 보장과 경제적 안정성 확보에 중요한 역할을 합니다",
            "summary": "노인 빈곤 문제 해결",
            "created_by": "user1",
            "created_at": "2025-05-12T11:30:00Z",
            "updated_at": "2025-05-12T11:30:00Z"
        }
    )
    
    # 세 번째 요청 - 근거가 추가된 트리, 1개 리뷰 요청
    # Third request - tree with added evidence, requesting 1 review
    print("\n==== 세 번째 요청: 근거가 추가된 트리, 1개 리뷰 요청 ====")
    payload3 = {
        "tree": tree_with_evidence,
        "review_num": 1,
        "student_id": student_id,
        "assignment_id": assignment_id
    }
    response3 = requests.post(url, json=payload3)
    result3 = response3.json()
    print(json.dumps(result3, ensure_ascii=False, indent=2))


# Example for /reset endpoint
def example_reset():
    student_id = "student001"
    assignment_id = "assignment001"
    url = f"{BASE_URL}/reset?student_id={student_id}&assignment_id={assignment_id}"
    
    print("\n==== Reset 요청: 특정 학생과 과제의 상태 초기화 ====")
    response = requests.post(url)
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))


# Example for /resetall endpoint
def example_resetall():
    url = f"{BASE_URL}/resetall"
    
    print("\n==== ResetAll 요청: 모든 상태 초기화 ====")
    response = requests.post(url)
    result = response.json()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    
if __name__ == "__main__":
    #example_chat()
    #example_summary()
    example_review()
    # Uncomment below to test reset endpoints
    # example_reset()
    # example_resetall()