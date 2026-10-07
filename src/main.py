import os
import sys

# Suppress Qt DPI warning on Windows before importing GUI or capture libs
os.environ["QT_LOGGING_RULES"] = "qt.qpa.window=false"

import base64
import io
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
from ping3 import ping

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    Qt,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMenu,
    QScrollArea,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

load_dotenv()

# --- Global Settings ---
SAMPLE_RATE = 16000
CHANNELS = 1

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.getenv("NEBIUS_API_KEY"),
)

CLASSIFIER_MODEL = "MiniMaxAI/MiniMax-M3"
TEXT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
VISION_MODEL = "zai-org/GLM-5.3-Flash"

VISION_SYSTEM_PROMPT = """
You are PortAI, a desktop AI assistant that analyzes screenshots.

Answer the user's question using only information that is clearly visible
in the screenshot.

Do not guess, infer, or invent information that is not visible.

If the requested information is partially visible, clearly state what you
can determine and what cannot be determined.

If the screenshot is insufficient to answer the question, say so.

Be concise and directly answer the user's question.
"""

conversation_history = []


# --- Core Backend Functions ---

def check_internet():
    response = ping("8.8.8.8", timeout=2)
    return response is not None


def get_vram_gb():
    try:
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return -1


def load_whisper_model():
    if torch.cuda.is_available():
        vram_gb = get_vram_gb()
        if 0 < vram_gb <= 3.0:
            model_name = "base"
        elif vram_gb > 3.0:
            model_name = "turbo"
        else:
            model_name = "base"
        try:
            supported_compute_types = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in supported_compute_types:
                compute_type = "float16"
            elif "int8_float16" in supported_compute_types:
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


def record_audio():
    audio_chunks = []

    def audio_callback(indata, _frames, _time_info, status):
        if status:
            print(status)
        audio_chunks.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback
    ):
        while keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt"):
            time.sleep(0.02)

    return audio_chunks


def transcribe_audio(whisper_model, audio_chunks):
    if not audio_chunks:
        return None

    audio_samples = np.concatenate(audio_chunks, axis=0).flatten().astype(np.float32)

    if len(audio_samples) < SAMPLE_RATE * 0.5:
        return None

    segments, _ = whisper_model.transcribe(
        audio_samples, beam_size=5, language="en", vad_filter=True
    )
    transcribed_text = "".join([segment.text for segment in segments]).strip()

    if transcribed_text:
        return transcribed_text
    return None


def classify_query_type(prompt, conversation_history):
    history_transcribed_text = ""
    for message in conversation_history:
        history_transcribed_text += f"{message['role']}: {message['content']}\n"

    classifier_prompt = f"""
You are a routing classifier.

Your ONLY job is to classify the LATEST USER MESSAGE.

Return EXACTLY ONE WORD:

TEXT
or
VISION
or
CLEARCONTEXT

CONVERSATION HISTORY:
{history_transcribed_text}

LATEST USER MESSAGE:
{prompt}

CLASSIFICATION RULES:

TEXT:
- The latest message can be answered using transcribed_text.
- General knowledge questions are TEXT.
- Coding questions are TEXT.
- Conversational questions are TEXT.
- Follow-up questions are TEXT if their meaning can be understood from the conversation history.

VISION:
- The latest message requires seeing the user's screen, window, UI, image, or other visual information.
- The user explicitly asks about something visible on their screen.
- The answer cannot be determined from the conversation history and requires visual information.

CLEARCONTEXT:
- The latest message is a query about clearing the conversation context.
- the user asks to clear the conversation history or context
- the user may ask you to forget the previous messages
- the user may ask you to reset the conversation

IMPORTANT:
- Use the conversation history ONLY to understand references such as "it", "that", "this", or "they".
- NEVER answer the latest user message.
- NEVER explain your classification.
- NEVER provide reasoning.
- Output ONLY TEXT or VISION or CLEARCONTEXT.
"""

    response = client.chat.completions.create(
        model=CLASSIFIER_MODEL,
        messages=[{"role": "user", "content": classifier_prompt}],
    )

    answer = response.choices[0].message.content.strip().upper()
    print(f"RAW CLASSIFIER OUTPUT: {repr(answer)}")
    if answer not in {"TEXT", "VISION", "CLEARCONTEXT"}:
        return "UNKNOWN"

    return answer


def generate_text_response(prompt):
    global conversation_history
    conversation_history.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Answer the question in a concise and informative manner.",
            },
            *conversation_history,
        ],
    )
    answer = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})
    return answer


def generate_vision_response(prompt):
    global conversation_history

    screenshot = pyautogui.screenshot()
    screenshot_buffer = io.BytesIO()
    screenshot.save(screenshot_buffer, format="PNG")
    screenshot_bytes = screenshot_buffer.getvalue()

    screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
    ss_url = f"data:image/png;base64,{screenshot_base64}"

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": ss_url}},
                ],
            },
        ],
    )
    answer = response.choices[0].message.content

    conversation_history.append({"role": "user", "content": prompt})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, screenshot_bytes


# --- Background Thread ---

class AssistantWorker(QThread):
    status_signal = pyqtSignal(str)
    response_signal = pyqtSignal(str, str, bytes)  # prompt, answer, screenshot_bytes

    def __init__(self):
        super().__init__()
        self.whisper_model = None
        self.is_running = True

    def run(self):
        if not check_internet():
            self.status_signal.emit("No internet access")
            return

        self.status_signal.emit("Loading Whisper...")
        self.whisper_model = load_whisper_model()
        self.status_signal.emit("Hold Ctrl + Alt to speak")

        while self.is_running:
            if keyboard.is_pressed("ctrl") and keyboard.is_pressed("alt"):
                self.status_signal.emit("Listening...")
                audio_chunks = record_audio()

                self.status_signal.emit("Transcribing...")
                prompt = transcribe_audio(self.whisper_model, audio_chunks)

                if not prompt:
                    self.status_signal.emit("No speech detected")
                    time.sleep(1)
                    self.status_signal.emit("Hold Ctrl + Alt to speak")
                    continue

                self.status_signal.emit(f"Thinking: {prompt}")
                question_type = classify_query_type(prompt, conversation_history)

                if question_type == "VISION":
                    answer, screenshot_bytes = generate_vision_response(prompt)
                    self.response_signal.emit(prompt, answer, screenshot_bytes)
                elif question_type == "TEXT":
                    answer = generate_text_response(prompt)
                    self.response_signal.emit(prompt, answer, b"")
                elif question_type == "CLEARCONTEXT":
                    conversation_history.clear()
                    self.response_signal.emit(
                        prompt, "Conversation context cleared.", b""
                    )
                else:
                    self.response_signal.emit(
                        prompt, "Unable to determine query type.", b""
                    )

                self.status_signal.emit("Hold Ctrl + Alt to speak")

            time.sleep(0.05)


# --- GUI Overlay Widgets ---

class DynamicIsland(QWidget):
    """Top bar island popping down from top center, stealth mode (no taskbar)."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.init_ui()
        self.reposition()

    def init_ui(self):
        self.container = QWidget(self)
        self.container.setStyleSheet("""
            QWidget {
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 18px;
            }
            QLabel {
                color: #CDD6F4;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                font-weight: 600;
            }
        """)

        layout = QHBoxLayout(self.container)
        layout.setContentsMargins(16, 8, 16, 8)

        self.dot = QLabel("●", self)
        self.dot.setStyleSheet("color: #A6E3A1; font-size: 14px;")
        layout.addWidget(self.dot)

        self.label = QLabel("PortAI Initializing...", self)
        layout.addWidget(self.label)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setYOffset(6)
        self.container.setGraphicsEffect(shadow)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)

    def reposition(self):
        screen = QApplication.primaryScreen().geometry()
        width = 340
        height = 46
        x = (screen.width() - width) // 2
        y = 12
        self.setGeometry(x, y, width, height)

    def set_status(self, text):
        self.label.setText(text)
        if "Listening" in text:
            self.dot.setStyleSheet("color: #F38BA8; font-size: 14px;")
        elif "Thinking" in text or "Transcribing" in text:
            self.dot.setStyleSheet("color: #FAB387; font-size: 14px;")
        else:
            self.dot.setStyleSheet("color: #A6E3A1; font-size: 14px;")


class SideDrawer(QWidget):
    """Side drawer sliding in from screen right, stealth mode (no taskbar)."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.drawer_width = 380
        self.drawer_height = 500

        self.init_ui()
        self.reposition_hidden()

    def init_ui(self):
        self.container = QWidget(self)
        self.container.setStyleSheet("""
            QWidget {
                background-color: #1E1E2E;
                border: 1px solid #313244;
                border-radius: 16px;
            }
            QLabel#header {
                color: #89B4FA;
                font-family: 'Segoe UI', sans-serif;
                font-size: 15px;
                font-weight: bold;
            }
            QLabel#prompt {
                color: #A6E3A1;
                font-size: 13px;
                font-weight: 600;
            }
            QTextEdit {
                background-color: #181825;
                color: #CDD6F4;
                border: 1px solid #313244;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
        """)

        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel("PortAI Assistant", self)
        header.setObjectName("header")
        layout.addWidget(header)

        self.prompt_label = QLabel("", self)
        self.prompt_label.setObjectName("prompt")
        self.prompt_label.setWordWrap(True)
        layout.addWidget(self.prompt_label)

        self.image_label = QLabel(self)
        self.image_label.hide()
        layout.addWidget(self.image_label)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none; background: transparent;")

        self.content_label = QLabel("", self)
        self.content_label.setWordWrap(True)
        self.content_label.setStyleSheet("color: #CDD6F4; font-size: 13px;")
        self.scroll_area.setWidget(self.content_label)

        layout.addWidget(self.scroll_area)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setXOffset(-4)
        shadow.setYOffset(4)
        self.container.setGraphicsEffect(shadow)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)

        self.anim = QPropertyAnimation(self, b"geometry")
        self.anim.setDuration(350)
        self.anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def reposition_hidden(self):
        screen = QApplication.primaryScreen().geometry()
        x = screen.width()
        y = (screen.height() - self.drawer_height) // 2
        self.setGeometry(x, y, self.drawer_width, self.drawer_height)

    def slide_in(self, prompt, response, screenshot_bytes=b""):
        screen = QApplication.primaryScreen().geometry()

        self.prompt_label.setText(f"You: {prompt}")
        self.content_label.setText(response)

        if screenshot_bytes:
            image = QImage.fromData(screenshot_bytes)
            pixmap = QPixmap.fromImage(image).scaledToWidth(
                340, Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            self.image_label.show()
        else:
            self.image_label.hide()

        start_x = screen.width()
        end_x = screen.width() - self.drawer_width - 20
        y = (screen.height() - self.drawer_height) // 2

        self.show()
        self.anim.setStartValue(QRect(start_x, y, self.drawer_width, self.drawer_height))
        self.anim.setEndValue(QRect(end_x, y, self.drawer_width, self.drawer_height))
        self.anim.start()

    def slide_out(self):
        screen = QApplication.primaryScreen().geometry()
        start_x = self.x()
        end_x = screen.width()
        y = self.y()

        self.anim.setStartValue(QRect(start_x, y, self.drawer_width, self.drawer_height))
        self.anim.setEndValue(QRect(end_x, y, self.drawer_width, self.drawer_height))
        self.anim.start()


# --- Main Application Controller ---

def create_tray_icon():
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setBrush(QColor("#89B4FA"))
    painter.drawEllipse(2, 2, 28, 28)
    painter.end()
    return QIcon(pixmap)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    island = DynamicIsland()
    island.show()

    drawer = SideDrawer()

    tray_icon = QSystemTrayIcon(create_tray_icon(), app)
    tray_menu = QMenu()

    toggle_drawer_action = QAction("Toggle Side Drawer", app)
    toggle_drawer_action.triggered.connect(
        lambda: drawer.slide_out()
        if drawer.x() < QApplication.primaryScreen().geometry().width()
        else drawer.slide_in("Manual Toggle", "PortAI is active.")
    )
    tray_menu.addAction(toggle_drawer_action)

    clear_context_action = QAction("Clear Context", app)
    clear_context_action.triggered.connect(lambda: conversation_history.clear())
    tray_menu.addAction(clear_context_action)

    tray_menu.addSeparator()

    exit_action = QAction("Exit PortAI", app)
    exit_action.triggered.connect(app.quit)
    tray_menu.addAction(exit_action)

    tray_icon.setContextMenu(tray_menu)
    tray_icon.setToolTip("PortAI Assistant (Ctrl + Alt)")
    tray_icon.show()

    worker = AssistantWorker()
    worker.status_signal.connect(island.set_status)
    worker.response_signal.connect(drawer.slide_in)
    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()