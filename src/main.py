import ctranslate2
import keyboard
import numpy as np
import sounddevice as sd
import time
import torch
from faster_whisper import WhisperModel
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)
"""
settings for the microphone
"""
SAMPLE_RATE = 16000
CHANNELS = 1

def gpuCheck():
    """Checks for which nvidia gpu is available

    Returns:
        bool: Returns True if Nvidia GPU exists else false
    """
    return torch.cuda.is_available()

def vramCheck():
    """Checks amount of vram

    Returns:
        int: Size of vram in GB
    """
    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (
            1024**3
        )
        return total_vram
    except Exception as e:
        return -1

def modelDecider():
    """Chooses models

    Returns:
        WhisperModel: Returns whispermodel based on hardware
    """
    if gpuCheck():
        vram = vramCheck()
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
            pass
    return WhisperModel("base", device="cpu", compute_type="int8")

def audioRecord():

    """
    Records the audio into chunks and then appends them together
    """

    audio_data = []

    def audioappend(indata, frames, time_info, status):
        if status:
            print(status)
        audio_data.append(indata.copy())

    # Explicitly using keyword arguments for safety
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audioappend
    ):
        while keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt"):
            time.sleep(0.02)

    return audio_data

def inputCheck():

    """
    Checks if both keys are pressed then runs the transcribeaudio function
    checks for key input 20 times a second
    """

    while True:
        if keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt"):
            break
        time.sleep(0.05)

def transcribeaudio(model, audio_data):
    """
    transcribes the audio in chunks and adds them together
    also returns null if no audio input is given
    """
    if not audio_data:
        return

    audio_np = np.concatenate(audio_data, axis=0).flatten().astype(np.float32)

    if len(audio_np) < SAMPLE_RATE * 0.5:
        return

    segments, info = model.transcribe(
        audio_np, beam_size=5, language="en", vad_filter=True
    )
    text = "".join([segment.text for segment in segments]).strip()

    if text:
        return text

def qTypeanalysis(prompt_):
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
                        "text": prompt_
                    }
                ]
            }
        ]
    )

    return (response.choices[0].message.content)

def main():
    model = modelDecider()
    print("Model loaded successfully.")
    try:
        while True:
            inputCheck()
            audio_data = audioRecord()
            print("DONE")
            prompt = transcribeaudio(model, audio_data)
            print(prompt)
            qType = qTypeanalysis(prompt)
            print(qType)
    except KeyboardInterrupt:
        print(f"EXIT\n{model}")


if __name__ == "__main__":
    main()