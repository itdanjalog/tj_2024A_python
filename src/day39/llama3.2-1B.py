import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


def chat_with_bot():
    print("챗봇에 질문하세요! (종료하려면 'exit' 입력)")

    while True:
        user_input = input("당신: ")

        if user_input.lower() == 'exit':
            print("챗봇을 종료합니다.")
            break

        # 챗봇 응답 생성
        prompt = f"User: {user_input}\nChatbot:"
        response = pipe(prompt, max_length=50, num_return_sequences=1)[0]['generated_text']

        # 응답 출력
        chatbot_response = response.split("Chatbot:")[-1].strip()  # 챗봇 응답 부분만 추출
        print(f"챗봇: {chatbot_response}")


# 챗봇 시작
chat_with_bot()