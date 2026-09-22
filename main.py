import whisper
import sounddevice as sd
import subprocess
import torch

def gpucheck():
    # check for available nvidia gpu
    try:
        subprocess.check_output(["nvidia-smi"])
        return True
    except FileNotFoundError:
        return False

def vramCheck():
    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return total_vram
    except Exception as e:
        return -1

def main():
    if (__name__ == "__main__"):

        if gpucheck():
            vram = vramCheck()
            match vram:
                case vram if vram <= 3.0:
                    model = whisper.load_model("small", device="cuda")
                case vram if vram <= 7.0:
                    model = whisper.load_model("turbo", device="cuda")
                case vram if vram <= 15.0:
                    model = whisper.load_model("large-v3", device="cuda")
                case _:
                    model = whisper.load_model("base", device="cuda")
        else:
            model = whisper.load_model("base", device="cpu")
main()