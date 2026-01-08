import requests
import json

# 测试输入
messages = [
    {"role": "system", "content": "你是一个知识库助手，请根据用户提供的文档回答问题。"},
    {"role": "user", "content": "根据文档内容，师徒四人获得的封号是什么？"}
]

# 发送请求
url = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "Qwen3-14B-AWQ",
    "messages": messages,
    "stream": False,
    "max_tokens": 1000,
    "temperature": 0.1
}

print("Sending request to /v1/chat/completions...")
print(f"Payload: {json.dumps(payload, ensure_ascii=False)}")

response = requests.post(url, json=payload)

print(f"\nResponse status code: {response.status_code}")
print(f"Response headers: {response.headers}")
print(f"Response content: {response.text}")

# 解析响应
if response.status_code == 200:
    try:
        data = response.json()
        if "choices" in data and data["choices"]:
            content = data["choices"][0]["message"]["content"]
            print(f"\nGenerated content: {content}")
            print(f"Content length: {len(content)} characters")
    except json.JSONDecodeError as e:
        print(f"\nJSON decode error: {e}")
        print(f"Raw response: {response.text}")
