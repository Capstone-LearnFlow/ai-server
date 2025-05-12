import requests

BASE_URL = "http://localhost:8000"

# Example for /chat endpoint
def example_chat():
    url = f"{BASE_URL}/chat"
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"}
        ],
        "model": "gpt-4.1-mini"
    }
    response = requests.post(url, json=payload)
    print("/chat response:", response.json())

# Example for /summary endpoint
def example_summary():
    url = f"{BASE_URL}/summary"
    payload = {
        "contents": [
            "Traders will get their first sense of the initial inflationary effects of the tariffs with the release of key inflation data this week. April’s Consumer Price Index (CPI) report is due Tuesday, followed by retail sales and the Producer Price Index (PPI) on Thursday.",
            "Brent (BZ=F) traded around $64 a barrel, paring an advance in early Asian trading after last week notching up its first weekly gain in three, while West Texas Intermediate (CL=F) was near $61. After negotiations in Geneva, US Treasury Secretary Scott Bessent and Trade Representative Jamieson Greer said that they were upbeat on progress and would share more information on Monday, with the positive sentiment echoed by their Chinese counterparts."
        ]
    }
    response = requests.post(url, json=payload)
    print("/summary response:", response.json())

# Example for /review endpoint
def example_review():
    url = f"{BASE_URL}/review"
    tree = {
        "id": "root",
        "type": "subject",
        "child": [
            {
                "id": "1",
                "type": "claim",
                "child": [],
                "sibling": [],
                "content": """최근에 정년 연장을 공론화한 곳은 행정안전부다. 이달 14일부터 행안부 소속 공무직 근로자의 정년이 60세에서 65세로 바뀌었다. 행안부 공무직은 기존 60세 정년을 맞은 해에 연장 신청을 하면 별도 심사를 거쳐 1964년생은 63세, 1965~1968년생은 64세, 1969년생부터는 65세로 정년이 늘어난다. 공무직은 국가나 지방자치단체에서 근무하는 민간 무기계약직 근로자다. 문재인 정부가 추진한 비정규직의 정규직화 과정에서 생겨난 직종으로 시설관리, 경비, 미화 등의 업무를 맡고 있다.

대구시도 비슷한 방식으로 공무직 정년을 연장했다. 내년에 60세가 되는 1965년생 근로자 정년을 61세로 늘린 뒤 순차적으로 확대해 2029년에 근로자 정년을 65세로 조정하기로 했다. 이미 서울시 산하 기초지방자치단체 등도 정년을 65세로 연장했다. 몇몇 중앙 부처도 청소업 등 일부 업종에 한해 정년을 65세로 바꿨다. 60세가 넘은 근로자를 계약직 등으로 재고용하는 사업장 비중이 지난해 36%로 역대 최고를 기록하기도 했다. 해외에서도 정년 연장 움직임이 활발하다. 독일과 프랑스는 연금 수급 개시 연령 이상으로 정년을 설정할 수 있게 했고, 미국과 영국에선 정년 자체가 존재하지 않는다.

정년 연장을 하면 기업 입장에선 숙련 근로자의 노하우를 잘 활용할 수 있게 된다. 젊은 직원들이 아무리 뛰어나도 30년 이상의 경험을 지닌 베테랑의 경륜을 넘어서긴 쉽지 않다. 고용 안정성을 강화해 근로 의욕을 높이는 동시에 노인 빈곤 문제를 해결할 방안이 될 수 있다.

생산인구가 감소하는 한국에선 정년 연장이 필수라는 시각도 있다. 950만 명이 넘는 2차 베이비붐 세대(1964~1974년생)가 올해부터 차례대로 정년을 맞는다. 올해 퇴직하는 1964년생은 국민연금을 63세부터 받는데 이렇게 되면 3년의 소득 공백이 발생한다.

2072년이 되면 생산인구(15~64세)는 2000만 명가량 급감한다. 이런 생산인구 절벽에 대응하기 위해 이중근 신임 대한노인회장은 노인 연령 기준을 65세에서 75세로 높이자고 제안했다.""",
                "summary": "AI impact on education",
                "created_by": "user1",
                "created_at": "2025-05-12T10:00:00Z",
                "updated_at": "2025-05-12T10:00:00Z"
            }
        ],
        "sibling": [],
        "content": "The future of technology",
        "summary": "Tech future",
        "created_by": "user1",
        "created_at": "2025-05-12T09:00:00Z",
        "updated_at": "2025-05-12T09:00:00Z"
    }
    payload = {
        "tree": tree,
        "review_request": "1"
    }
    response = requests.post(url, json=payload)
    print("/review response:", response.json())

if __name__ == "__main__":
    example_chat()
    example_summary()
    example_review()
