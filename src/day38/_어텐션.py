import torch
from transformers import BertTokenizer, BertForSequenceClassification

# KoBERT 모델과 토크나이저 로드
model_name = "monologg/kobert"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# 모델을 평가 모드로 설정
model.eval()

# 질문-응답 데이터셋 (예시)
responses = {
    0: "안녕하세요! 무엇을 도와드릴까요?",  # 클래스 0에 대한 응답
    1: "오늘 날씨는 맑습니다.",  # 클래스 1에 대한 응답
    2: "천만에요! 또 궁금한 점이 있으면 물어보세요.",  # 클래스 2에 대한 응답
    3: "저는 항상 좋습니다! 당신은요?",  # 클래스 3에 대한 응답
    4: "저는 클라우드에 살고 있어요.",  # 클래스 4에 대한 응답
    5: "이런 질문은 잘 모르겠어요."  # 클래스 5에 대한 응답
}


# 챗봇 함수 정의
def chatbot_response(user_input):
    # 입력을 인코딩
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)

    # 예측 수행
    with torch.no_grad():
        logits = model(**inputs).logits

    # 예측된 클래스 인덱스
    predicted_class = logits.argmax().item()

    # 예측한 클래스에 따른 응답 생성
    return responses.get(predicted_class, "이런 질문은 잘 모르겠어요.")


# 챗봇과의 대화 시작
print("챗봇에 질문하세요! (종료하려면 '종료' 입력)")

while True:
    user_input = input("사용자: ")
    if user_input == "종료":
        print("챗봇: 안녕히 가세요!")
        break
    response = chatbot_response(user_input)
    print(f"챗봇: {response}")
