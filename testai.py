import os
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-M3",
     messages=[
        {
            "role": "system",
            "content": """
            You are a lightweight intent classification engine. Your sole task is to analyze incoming user questions and classify whether answering them requires visual context (a screen capture/image) or purely text/system processing.

## CLASSIFICATION RULES

1. **CLASSIFY AS VISION IF:**
   - The user explicitly mentions looking at, reading, or analyzing the screen, UI, display, window, image, layout, or visual elements.
   - Information required to answer the question is missing, ambiguous, or incomplete, and could be resolved by viewing the current display state. **Always default to `VISION` when in doubt.**

2. **CLASSIFY AS TEXT ONLY IF:**
   - The question is fully self-contained, theoretical, code-only, conversational, or a direct system/CLI command with no missing contextual details.

## OUTPUT FORMAT

Respond ONLY with a JSON object in this exact schema. Do not include introductory text, explanations, or Markdown blocks outside the JSON:


ONLY RESPOND WITH VISION OR TEXT dont add classification or anything just the two words TEXT or VISION
            """
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "whats wrong with this code"
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)