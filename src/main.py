import sounddevice as sd
import torch
import ctranslate2
from faster_whisper import WhisperModel
import numpy as np

<<<<<<< Updated upstream
audioBuffer = np.zeros(0, dtype=np.float32)
=======
"""
settings for the microphone
"""
SAMPLE_RATE = 16000
CHANNELS = 1
>>>>>>> Stashed changes

def gpuCheck():
    """Checks for which nvidia gpu is available

    Returns:
        bool: Returns True if Nvidia GPU exists else false
    """
    return torch.cuda.is_available() and (ctranslate2.get_cuda_device_count() > 0)

def vramCheck():
    """Checks amount of vram

    Returns:
        int: Size of vram
    """
    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
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
        if 0<vram<=3.0:
            model_name = "base"
        elif vram>3.0:
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
<<<<<<< Updated upstream
            print(f"Loading {model_name} on CUDA")
            return WhisperModel(model_name, device="cuda",compute_type=compute_type)
        
        except Exception as e:
            print(f"GPU initialization failed: {e}")
    print("Running on CPU")
    return WhisperModel("base", device="cpu", compute_type="int8")

def audioRecord(model):
    global audioBuffer
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_DURATION = 5
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
    print("🎤 Listening... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=audioCallback):
            while True:
                if len(audioBuffer) >= CHUNK_SIZE:
                    current_chunk = audioBuffer[:CHUNK_SIZE]
                    audioBuffer = audioBuffer[CHUNK_SIZE:]
                    
                    segments, info = model.transcribe(current_chunk, beam_size=5)
                    
                    for segment in segments:
                        if segment.text.strip():
                            print(f"[{info.language}] {segment.text}")
                            
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        

def audioCallback(indata, frames, time, status):
    global audioBuffer
    if status:
        print(f"status warning: {status}")
    audioBuffer = np.append(audioBuffer, indata.ravel())
    
=======
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
>>>>>>> Stashed changes

def main():
    model = modelDecider()
    print("Model loaded successfully.")
<<<<<<< Updated upstream
    audioRecord(model)
=======
    try:
        while True:
            inputCheck()
            audio_data = audioRecord()
            print("DONE")
            prompt = transcribeaudio(model, audio_data)
            print(prompt)
    except KeyboardInterrupt:
        print(f"EXIT\n{model}")
>>>>>>> Stashed changes

if (__name__ == "__main__"):
    main()