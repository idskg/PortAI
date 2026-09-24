import sounddevice as sd
import subprocess
import torch
from faster_whisper import WhisperModel

def gpucheck():
    """Checks for which nvidia gpu is available

    Returns:
        bool: Returns True if Nvidia GPU exists else false
    """
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except FileNotFoundError:
        return False

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
    if gpucheck():
        vram = vramCheck()
        match vram:
            case vram if vram <= 3.0:
                model = WhisperModel("base", device="cuda", compute_type="int8_float16")
            case vram if vram >= 3.001:
                model = WhisperModel("turbo", device="cuda", compute_type="float16")
            case _:
                model = WhisperModel("base", device="cuda", compute_type="float16")
    else:
        model = WhisperModel("base", device="cpu", compute_type="int8")
    return model

def main():
    if (__name__ == "__main__"):
        model = modelDecider()
        print("Model loaded successfully.")
        

main()