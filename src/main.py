import sounddevice as sd
import torch
import ctranslate2
from faster_whisper import WhisperModel

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
            print(f"Loading {model_name} on CUDA")
            return WhisperModel(model_name, device="cuda",compute_type=compute_type)
        
        except Exception as e:
            print(f"GPU initialization failed: {e}")
    print("Running on CPU")
    return WhisperModel("base", device="cpu", compute_type="int8")
    

def main():
    model = modelDecider()
    print("Model loaded successfully.")
        

if (__name__ == "__main__"):
    main()