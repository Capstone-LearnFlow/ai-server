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
    
    # 첫 번째 요청 - 기본 1개 리뷰
    # First request - default 1 review
    print("\n==== 첫 번째 요청: 초기 트리, 기본 1개 리뷰 ====")
    payload1 = {
        "tree": initial_tree
    }
    response1 = requests.post(url, json=payload1)
    result1 = response1.json()
    print(json.dumps(result1, ensure_ascii=False, indent=2))
    
    # 두 번째 요청 - 동일한 트리, 3개 리뷰 요청
    # Second request - same tree, requesting 3 reviews
    print("\n==== 두 번째 요청: 동일한 트리, 3개 리뷰 요청 ====")
    payload2 = {
        "tree": initial_tree,
        "review_num": 3
    }
    response2 = requests.post(url, json=payload2)
    result2 = response2.json()
    print(json.dumps(result2, ensure_ascii=False, indent=2))
    
    # 트리에 새로운 근거 노드 추가
    # Add new evidence node to the tree
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
    
    # 세 번째 요청 - 근거가 추가된 트리, 2개 리뷰 요청
    # Third request - tree with added evidence, requesting 2 reviews
    print("\n==== 세 번째 요청: 근거가 추가된 트리, 2개 리뷰 요청 ====")
    payload3 = {
        "tree": tree_with_evidence,
        "review_num": 2
    }
    response3 = requests.post(url, json=payload3)
    result3 = response3.json()
    print(json.dumps(result3, ensure_ascii=False, indent=2))
    
    # 트리에 새로운 반론 노드와 그에 대한 근거 추가
    # Add a counterargument and its evidence to the tree
    tree_with_counterargument = json.loads(json.dumps(tree_with_evidence))  # Deep copy
    tree_with_counterargument["child"][0]["child"].append({
        "id": "counter1",
        "type": "반론",
        "child": [
            {
                "id": "counter_evidence1",
                "type": "근거",
                "child": [],
                "sibling": [],
                "content": "정년 연장은 청년 일자리를 감소시키는 결과를 초래할 수 있다. 일본의 사례에서 보듯이 정년 연장 후 청년 실업률이 증가하는 경향이 나타났다.",
                "summary": "청년 일자리 감소 우려",
                "created_by": "user2",
                "created_at": "2025-05-12T13:00:00Z",
                "updated_at": "2025-05-12T13:00:00Z"
            }
        ],
        "sibling": [],
        "content": "정년 연장은 세대 간 일자리 갈등을 심화시킬 수 있으며, 조직의 신진대사를 저해할 우려가 있다.",
        "summary": "정년 연장의 부작용",
        "created_by": "user2",
        "created_at": "2025-05-12T12:45:00Z",
        "updated_at": "2025-05-12T12:45:00Z"
    })
    
    # 네 번째 요청 - 반론이 추가된 트리, 2개 리뷰 요청
    # Fourth request - tree with added counterargument, requesting 2 reviews
    print("\n==== 네 번째 요청: 반론이 추가된 트리, 2개 리뷰 요청 ====")
    payload4 = {
        "tree": tree_with_counterargument,
        "review_num": 2
    }
    response4 = requests.post(url, json=payload4)
    result4 = response4.json()
    print(json.dumps(result4, ensure_ascii=False, indent=2))
    
    # 반론 노드에 대한 반론 추가 (반론의 반론)
    # Add a counterargument to the counterargument
    tree_with_nested_argument = json.loads(json.dumps(tree_with_counterargument))  # Deep copy
    counter_node = next((node for node in tree_with_nested_argument["child"][0]["child"] if node["id"] == "counter1"), None)
    if counter_node:
        counter_node["child"].append({
            "id": "counter_to_counter1",
            "type": "반론",
            "child": [
                {
                    "id": "counter_to_counter_evidence1",
                    "type": "근거",
                    "child": [],
                    "sibling": [],
                    "content": "정년 연장과 청년 일자리는 상충관계가 아닌 보완관계가 될 수 있다. 고령자의 경험과 청년의 새로운 아이디어가 결합하면 기업 경쟁력이 향상된다는 연구 결과가 있다.",
                    "summary": "세대 간 상생 가능성",
                    "created_by": "user1",
                    "created_at": "2025-05-12T14:30:00Z",
                    "updated_at": "2025-05-12T14:30:00Z"
                }
            ],
            "sibling": [],
            "content": "정년 연장과 청년 고용은 상충관계가 아니며, 세대 간 지식 전수와 멘토링을 통해 생산성을 높일 수 있다.",
            "summary": "세대 간 상생 가능성",
            "created_by": "user1",
            "created_at": "2025-05-12T14:15:00Z",
            "updated_at": "2025-05-12T14:15:00Z"
        })
    
    # 다섯 번째 요청 - 복잡한 중첩 논증 구조, 2개 리뷰 요청
    # Fifth request - complex nested argumentation structure, requesting 2 reviews
    print("\n==== 다섯 번째 요청: 복잡한 중첩 논증 구조, 2개 리뷰 요청 ====")
    payload5 = {
        "tree": tree_with_nested_argument,
        "review_num": 2
    }
    response5 = requests.post(url, json=payload5)
    result5 = response5.json()
    print(json.dumps(result5, ensure_ascii=False, indent=2))
    
if __name__ == "__main__":
    #example_chat()
    #example_summary()
    example_review()