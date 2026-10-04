import base64
import io
import json
import os
import time

import ctranslate2
import keyboard
import numpy as np
import pyautogui
import sounddevice as sd
import torch
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from openai import OpenAI

load_dotenv()

# settings for the microphone

SAMPLE_RATE = 16000
CHANNELS = 1

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY")
)

CLASSIFIER_MODEL = "MiniMaxAI/MiniMax-M3"
TEXT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
VISION_MODEL = None

conversation_history = []

def vram_check():
    """Returns the total VRAM of the first CUDA device in GB.

    Returns:
        float: Total VRAM in GB, or -1 if no CUDA device is available.
    """
    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )
        return total_vram
    except Exception:
        return -1

def model_decider():
    """
    Select and initialize a Whisper model based on available hardware.

    Uses CUDA when available and selects the model size based on VRAM.
    Falls back to the CPU with INT8 quantization if CUDA initialization
    fails or CUDA is unavailable.

    Returns:
        WhisperModel: Initialized Faster-Whisper model.
    """
    if torch.cuda.is_available():
        vram = vram_check()
        if 0 < vram <= 3.0:
            model_name = "base"
        elif vram > 3.0:
            model_name = "turbo"
        else:
            model_name = "base"
        try:
            supported_types = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in supported_types:
                compute_type = "float16"
            elif "int8_float16" in supported_types:
                compute_type = "int8_float16"
            else:
                compute_type = "float32"

            return WhisperModel(
                model_name, device="cuda", compute_type=compute_type
            )

        except Exception as e:
            print(f"CUDA initialization failed: {e}")
            print("Falling back to CPU.")
    return WhisperModel("base", device="cpu", compute_type="int8")

def audio_record():
    """Record microphone input while Ctrl+Alt are held.

    Audio is captured in chunks through a sounddevice InputStream.
    Recording stops when either Ctrl or Alt is released.

    Returns:
        list[np.ndarray]: Recorded audio chunks.
    """

    audio_data = []

    def audio_callback(indata, _frames, _time_info, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    # Explicitly using keyword arguments for safety
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback
    ):
        while keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt"):
            time.sleep(0.02)

    return audio_data

def input_check():
    """Block until Ctrl and Alt are pressed simultaneously."""

    while True:
        if keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt"):
            break
        time.sleep(0.05)

def transcribe_audio(whisper_model, audio_data):
    """Transcribe recorded audio using Faster-Whisper.

    Audio shorter than 0.5 seconds is ignored. Voice activity
    detection is enabled to filter out non-speech segments.

    Args:
        whisper_model (WhisperModel): Initialized Faster-Whisper model.
        audio_data (list[np.ndarray]): Recorded audio chunks.

    Returns:
        str | None: Transcribed text, or None if no usable speech
        was detected.
    """ 

    if not audio_data:
        return

    audio_np = np.concatenate(audio_data, axis=0).flatten().astype(np.float32)

    if len(audio_np) < SAMPLE_RATE * 0.5:
        return

    segments, _ = whisper_model.transcribe(
        audio_np, beam_size=5, language="en", vad_filter=True
    )
    text = "".join([segment.text for segment in segments]).strip()

    if text:
        return text

def analyze_question_type(prompt, conversation_history):
    """Classify a user prompt as requiring text or visual context.

    Sends the transcribed prompt to the intent-classification model,
    which returns either ``TEXT`` or ``VISION``.

    Args:
        prompt (str): Transcribed user question.

    Returns:
        tuple[str, str]: Classification and model reasoning.
    """
    history_text = ""

    for message in conversation_history:
        history_text += f"{message['role']}: {message['content']}\n"

    classifier_prompt = f"""
You are a routing classifier.

Your ONLY job is to classify the LATEST USER MESSAGE.

Return EXACTLY ONE WORD:

TEXT
or
VISION

CONVERSATION HISTORY:
{history_text}

LATEST USER MESSAGE:
{prompt}

CLASSIFICATION RULES:

TEXT:
- The latest message can be answered using text.
- General knowledge questions are TEXT.
- Coding questions are TEXT.
- Conversational questions are TEXT.
- Follow-up questions are TEXT if their meaning can be understood from the conversation history.

VISION:
- The latest message requires seeing the user's screen, window, UI, image, or other visual information.
- The user explicitly asks about something visible on their screen.
- The answer cannot be determined from the conversation history and requires visual information.

IMPORTANT:
- Use the conversation history ONLY to understand references such as "it", "that", "this", or "they".
- NEVER answer the latest user message.
- NEVER explain your classification.
- NEVER provide reasoning.
- Output ONLY TEXT or VISION.
"""

    response = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[
            {
                "role": "user",
                "content": classifier_prompt
            }
        ]
    )
    
    answer = response.choices[0].message.content.strip().upper()
    print(f"RAW CLASSIFIER OUTPUT: {repr(answer)}")
    if answer not in {"TEXT", "VISION"}:
        return "UNKNOWN"

    return answer

def generate_text_response(prompt):
    """Generate a response to a text-only user query.

    Args:
        prompt (str): User's transcribed question.

    Returns:
        tuple[str, str]: Generated answer and model reasoning.
    """

    print("Processing text based stuff")

    global conversation_history

    conversation_history.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer the question in a consice and informative manner."
            },
            *conversation_history
        ]
    )
    answer = response.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer

def generate_vision_response(prompt):

    screenshot = pyautogui.screenshot()

    ss_ram = io.BytesIO()
    screenshot.save(ss_ram, format="PNG")
    ss_ram.seek(0)

    ss_base64 = base64.b64encode(ss_ram.read()).decode("utf-8")
    ss_url = f"data:image/png;base64,{ss_base64}"

    response = client.chat.completions.create(
    model="zai-org/GLM-5.3-Flash",
    messages=[
        {
            "role": "system",
            "content": """SYSTEM_PROMPT"""
        },
        {
            "role": "user",
            "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url" : ss_url
                        }
                    }
                ]
            }
        ]
    )
    return (response.choices[0].message.content), ("1")

def main():
    whisper_model = model_decider()
    print("Model loaded successfully.")
    try:
        while True:
            input_check()
            audio_data = audio_record()
            print("DONE")
            prompt = transcribe_audio(whisper_model, audio_data)
            if not prompt:
                print("No speech detected.")
                continue
            else:
                print(f"{prompt} \n")

            question_type = analyze_question_type(prompt,conversation_history)
            print(question_type)
            answer, answer_reasoning = "", ""
            if(question_type == "VISION"):
                answer, answer_reasoning = generate_vision_response(prompt)
                print(answer)
                print(answer_reasoning)
            elif(question_type == "TEXT"):
                answer = generate_text_response(prompt)
                print(f"{answer} \n")
                # print(answer_reasoning)
            else:
                print("Unable to determine question type.")

    except KeyboardInterrupt:
        print(f"EXIT\n{whisper_model}")


if __name__ == "__main__":
    main()