import os
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

response = client.chat.completions.create(
    model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
     messages=[
        {
            "role": "system",
            "content": "Answer the question in a concise and informative manner.while keeping the context of the previous questions."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "what was my last question?"
                }
            ]
        }
    ]
)
answer_json = response.to_json()
answer_json = json.loads(answer_json)
answer = answer_json["choices"][0]["message"]["content"]
answer_reasoning = answer_json["choices"][0]["message"]["reasoning"]
print(answer)
print(answer_reasoning)