import torch
import customtkinter
from PIL import Image
import threading
import time
import random
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from gtts import gTTS
import tempfile
import pygame

customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("green")

pygame.mixer.init()

# suono per la fine
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        filename = fp.name

    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    try:
        os.remove(filename)
    except PermissionError:
        pass

# parola casuale
def get_random_word(file_path="words.txt"):
    if not os.path.exists(file_path):
        return "default"
    with open(file_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    if not words:
        return "default"
    return random.choice(words)

# thread gpt2
def gpt2_worker(prompt, duration_minutes, stop_event, buffer, lock):
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device)

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    process = psutil.Process(os.getpid())
    cores = psutil.cpu_count()
    start_time = time.time()

    with torch.no_grad():
        while time.time() - start_time < duration_minutes * 60:
            if stop_event.is_set():
                break

            temp = random.uniform(0.7, 1.3)
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=40,
                temperature=temp,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )

            text_out = tokenizer.decode(output[0], skip_special_tokens=True)
            snippet = text_out[-200:].replace("\n", " ")

            time.sleep(random.uniform(0.3, 0.8))

            cpu_percent_single_core = process.cpu_percent(interval=0.1)
            cpu_percent_total = min(cpu_percent_single_core / cores * 100, 100)

            with lock:
                buffer.append(cpu_percent_total)
                if len(buffer) > 120:
                    buffer.pop(0)

# start meditation
def start_meditation(entry_widget, text_widget, canvas, buffer, status_label=None, app=None):
    prompt = entry_widget.get().strip()
    if not prompt:
        return
    stop_event = threading.Event()
    lock = threading.Lock()

    if status_label:
        status_label.configure(
            text=f"Meditation on '{prompt}' started",
            text_color="#00ff99"  # verde
        )

    if app:
        app.fade_out = False

    def run_meditation():
        gpt2_worker(prompt, 10, stop_event, buffer, lock)

        if status_label:
            status_label.after(0, lambda: status_label.configure(
                text=f"The meditation on '{prompt}' is finished. We both thought, but you’re the one in charge of producing an output. Write your reflections down on a piece of paper.",
                text_color="#ff5555"  # rosso
            ))
        speak(f"Time is up! The meditation on {prompt} is finished. We both thought, but you’re the one in charge of producing an output. Write your reflections down on a piece of paper.")

        if app:
            app.fade_out = True

        # ripeschiamo una parola dal file
        new_word = get_random_word("words.txt")
        entry_widget.configure(state="normal")
        entry_widget.delete(0, "end")
        entry_widget.insert(0, new_word)
        entry_widget.configure(state="disabled")

    threading.Thread(target=run_meditation, daemon=True).start()

# parte dell'oscilloscopio
def draw_scale(canvas, canvas_height):
    canvas.delete("scale")
    width = canvas.winfo_width()
    if width <= 1:
        canvas.after(50, lambda: draw_scale(canvas, canvas_height))
        return
    margin_top = 5
    margin_bottom = 5
    line_length = 8
    for i in range(0, 101, 25):
        y = margin_top + (canvas_height - margin_top - margin_bottom) * (1 - i/100)
        canvas.create_line(width - line_length - 2, y, width - 2, y, fill="#888", width=1, tags="scale")
        canvas.create_text(width - line_length - 6, y, text=f"{i}%", anchor="e",
                           fill="#aaa", font=("Helvetica", 9), tags="scale")

def draw_wave(buffer, canvas, canvas_height, lock, fade_out=False):
    canvas.delete("wave")
    width = canvas.winfo_width()
    if width <= 1 or not buffer:
        return
    margin_right = 70
    usable_width = width - margin_right
    with lock:
        step = usable_width / max(len(buffer)-1, 1)
        smoothed = []
        alpha = 0.3
        for i, val in enumerate(buffer):
            if i == 0:
                smoothed.append(val)
            else:
                smoothed.append(smoothed[-1]*(1-alpha) + val*alpha)

        if fade_out:
            for i in range(len(smoothed)):
                smoothed[i] *= 0.95
                buffer[i] = smoothed[i]

        for i in range(1, len(smoothed)):
            x0 = (i-1) * step
            x1 = i * step
            y0 = canvas_height * (1 - smoothed[i-1]/100)
            y1 = canvas_height * (1 - smoothed[i]/100)
            canvas.create_line(x0, y0, x1, y1, fill="white", width=2, tags="wave")

def cpu_monitor_canvas(canvas, buffer, canvas_height=150, app=None):
    lock = threading.Lock()
    def update_loop():
        draw_scale(canvas, canvas_height)
        draw_wave(buffer, canvas, canvas_height, lock, fade_out=(app.fade_out if app else False))
        canvas.after(50, update_loop)
    update_loop()

# GUI
class MeditationApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        icon_path = os.path.join(os.path.dirname(__file__), "icona.ico")
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)

        self.title("10 Minutes Meditative Machine")
        self.geometry("1100x580")
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        self.buffer = [0]*120
        self.fade_out = False

        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        if os.path.exists(logo_path):
            from customtkinter import CTkImage
            logo_img = Image.open(logo_path).resize((140, 77))
            logo_ctk = CTkImage(light_image=logo_img, dark_image=logo_img, size=(140,77))
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_ctk, text="")
            self.logo_label.image = logo_ctk
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20,10))

        self.text_frame = customtkinter.CTkFrame(self, corner_radius=10)
        self.text_frame.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20,20), pady=(20,20), sticky="nsew")
        self.text_frame.grid_propagate(False)
        self.text_inner_frame = customtkinter.CTkFrame(self.text_frame, corner_radius=0, fg_color="#3B3B3B")
        self.text_inner_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.title_label = customtkinter.CTkLabel(
            self.text_inner_frame,
            text="Welcome to the 10 Minutes Meditative Machine.",
            font=("Helvetica", 14),
            text_color="#FFFFFF",
            fg_color="#3B3B3B",
            anchor='w',
            justify="left"
        )
        self.title_label.pack(fill="x", padx=12, pady=(5,10))

        # parte textbox con istruzioni
        self.text_box = customtkinter.CTkTextbox(self.text_inner_frame, corner_radius=10, font=("Helvetica", 13))
        self.text_box.pack(fill="both", expand=True)
        self.text_box.insert("end",
            "This is a useless linguistic machine.\n"
            "Your brain and my processor have to meditate\n"
            "for ten minutes straight on a prompt I impose.\n"
            "I will signal the end. Prepare to manifest an output\n\n\n"
            "Processor activity involved in word elaboration:"
        )
        self.text_box.tag_config("progress", foreground="#40c0ff")

        # textbox con la parola casuale
        random_word = get_random_word("words.txt")
        self.entry = customtkinter.CTkEntry(self, font=("Helvetica", 13))
        self.entry.insert(0, random_word)
        self.entry.configure(state="disabled")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20,0), pady=(20,20), sticky="nsew")

        self.status_label = customtkinter.CTkLabel(
            self.text_frame,
            text="",
            text_color="white",
            font=("Helvetica", 11, "italic"),
            anchor="w"
        )
        self.status_label.pack(side="bottom", anchor="w", padx=5, pady=(0,5))

        self.start_button = customtkinter.CTkButton(
            self,
            text="Start",
            command=lambda: start_meditation(
                self.entry, self.text_box, self.canvas, self.buffer,
                status_label=self.status_label, app=self
            )
        )
        self.start_button.grid(row=3, column=3, padx=(10,20), pady=(20,20), sticky="nsew")

        self.canvas_height = 150
        self.canvas = customtkinter.CTkCanvas(self.text_frame, width=1, height=self.canvas_height, bg="#3B3B3B", highlightthickness=0)
        self.canvas.place(relx=0.01, rely=0.4, relwidth=0.98)
        self.canvas.bind("<Configure>", lambda e: draw_scale(self.canvas, self.canvas_height))

        cpu_monitor_canvas(self.canvas, self.buffer, self.canvas_height, app=self)


if __name__ == "__main__":
    app = MeditationApp()
    app.mainloop()
