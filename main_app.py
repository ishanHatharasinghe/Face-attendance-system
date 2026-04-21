import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog
from tkcalendar import DateEntry
import cv2
import face_recognition
import numpy as np
import pickle
import pandas as pd
import os
import shutil
from datetime import datetime
import hashlib
import sys
from PIL import Image, ImageTk
import threading
import math
import pyttsx3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ─────────────────────────────────────────────
#  TEXT-TO-SPEECH  –  threaded so UI never freezes
# ─────────────────────────────────────────────
def speak(text):
    """Speak *text* in a daemon thread to keep Tkinter fully responsive."""
    def _run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

# ─────────────────────────────────────────────
#  CLAHE PRE-PROCESSOR
# ─────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(frame_bgr):
    """
    Enhance contrast via CLAHE on the L-channel of LAB colour space.
    Helps face detection under low-light or harsh sunlight.
    Returns the enhanced BGR frame.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# ─────────────────────────────────────────────
#  DESIGN TOKENS  –  Pure Black & White Monochrome
# ─────────────────────────────────────────────
BG          = "#080808"
BG2         = "#0D0D0D"
CARD        = "#111111"
CARD2       = "#181818"
CARD3       = "#202020"
BORDER      = "#2A2A2A"
BORDER2     = "#333333"

ACCENT      = "#FFFFFF"
ACCENT_DIM  = "#CCCCCC"
ACCENT2     = "#EEEEEE"
ACCENT2_DIM = "#AAAAAA"
ACCENT3     = "#DDDDDD"
WARN        = "#BBBBBB"
WARN_DIM    = "#888888"
INFO        = "#CCCCCC"

TEXT        = "#F0F0F0"
TEXT2       = "#BBBBBB"
MUTED       = "#555555"
MUTED2      = "#333333"

FONT_HEAD   = ("Consolas", 22, "bold")
FONT_TITLE  = ("Consolas", 10, "bold")
FONT_BODY   = ("Segoe UI", 9)
FONT_BODY2  = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 8)
FONT_TINY   = ("Segoe UI", 7)
FONT_BTN    = ("Segoe UI", 9, "bold")
FONT_MONO   = ("Consolas", 9)
FONT_MONO_S = ("Consolas", 8)

recent_logs = []

# ─────────────────────────────────────────────
#  ANIMATION HELPERS
# ─────────────────────────────────────────────
class Animator:
    @staticmethod
    def lerp(a, b, t):
        return a + (b - a) * t

    @staticmethod
    def ease_out(t):
        return 1 - (1 - t) ** 3

    @staticmethod
    def ease_in_out(t):
        if t < 0.5:
            return 4 * t * t * t
        return 1 - (-2 * t + 2) ** 3 / 2

def animate_label_fade(widget, text, color=ACCENT, steps=12):
    colors_from = MUTED
    r1, g1, b1 = int(colors_from[1:3], 16), int(colors_from[3:5], 16), int(colors_from[5:7], 16)
    r2, g2, b2 = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    widget.config(text=text)
    def step(i=0):
        if i >= steps:
            widget.config(fg=color)
            return
        t = Animator.ease_out(i / steps)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        widget.config(fg=f"#{r:02x}{g:02x}{b:02x}")
        widget.after(16, lambda: step(i + 1))
    step()

def pulse_widget(widget, color1, color2, times=3, interval=200):
    def toggle(n=0):
        if n >= times * 2:
            try:
                widget.config(highlightbackground=color1)
            except:
                pass
            return
        try:
            widget.config(highlightbackground=color2 if n % 2 == 0 else color1)
        except:
            pass
        widget.after(interval, lambda: toggle(n + 1))
    toggle()

# ─────────────────────────────────────────────
#  TOAST NOTIFICATION
# ─────────────────────────────────────────────
_toast_window = None

def show_toast(message, color=ACCENT, duration=2500):
    global _toast_window
    if _toast_window:
        try:
            _toast_window.destroy()
        except:
            pass

    toast = tk.Toplevel(root)
    toast.overrideredirect(True)
    toast.configure(bg=CARD2)
    toast.attributes("-topmost", True)
    toast.attributes("-alpha", 0.0)

    root.update_idletasks()
    rx, ry = root.winfo_x(), root.winfo_y()
    rw, rh = root.winfo_width(), root.winfo_height()

    tw, th = 320, 52
    target_x = rx + rw - tw - 20
    target_y = ry + rh - th - 20
    toast.geometry(f"{tw}x{th}+{target_x}+{target_y + 40}")

    border_f = tk.Frame(toast, bg=color, padx=2, pady=2)
    border_f.pack(fill="both", expand=True)
    inner = tk.Frame(border_f, bg=CARD2)
    inner.pack(fill="both", expand=True)

    dot = tk.Label(inner, text="●", fg=color, bg=CARD2, font=("Segoe UI", 10))
    dot.pack(side="left", padx=(12, 6))
    tk.Label(inner, text=message, fg=TEXT, bg=CARD2,
             font=("Segoe UI", 9, "bold"), anchor="w").pack(side="left", fill="x", expand=True)

    _toast_window = toast

    def fade_in(step=0):
        if step > 15:
            fade_out_later()
            return
        alpha = Animator.ease_out(step / 15)
        offset = int((1 - alpha) * 30)
        toast.attributes("-alpha", min(alpha, 0.95))
        toast.geometry(f"{tw}x{th}+{target_x}+{target_y + offset}")
        toast.after(16, lambda: fade_in(step + 1))

    def fade_out_later():
        toast.after(duration, fade_out)

    def fade_out(step=0):
        if step > 15:
            try:
                toast.destroy()
            except:
                pass
            return
        alpha = 0.95 * (1 - Animator.ease_out(step / 15))
        toast.attributes("-alpha", max(alpha, 0))
        toast.after(16, lambda: fade_out(step + 1))

    fade_in()

# ─────────────────────────────────────────────
#  CUSTOM WIDGETS
# ─────────────────────────────────────────────
class GlowButton(tk.Canvas):
    def __init__(self, parent, text, command=None, bg_color=ACCENT,
                 fg_color=BG, font=FONT_BTN, width=220, height=40,
                 radius=10, icon="", hover_color=None, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=parent.cget("bg"), highlightthickness=0, **kwargs)
        self.command   = command
        self.bg_color  = bg_color
        self.hover_col = hover_color or self._lighten(bg_color, 30)
        self.press_col = self._darken(bg_color, 20)
        self.fg_color  = fg_color
        self.radius    = radius
        self.label     = (icon + "  " + text).strip() if icon else text
        self.font      = font
        self.w         = width
        self.h         = height
        self._anim_color = bg_color

        self._draw(self.bg_color)
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _hex_to_rgb(self, h):
        return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)

    def _rgb_to_hex(self, r, g, b):
        return f"#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}"

    def _lighten(self, h, amt=30):
        r, g, b = self._hex_to_rgb(h)
        return self._rgb_to_hex(r+amt, g+amt, b+amt)

    def _darken(self, h, amt=20):
        r, g, b = self._hex_to_rgb(h)
        return self._rgb_to_hex(r-amt, g-amt, b-amt)

    def _lerp_color(self, c1, c2, t):
        r1, g1, b1 = self._hex_to_rgb(c1)
        r2, g2, b2 = self._hex_to_rgb(c2)
        t = Animator.ease_out(t)
        return self._rgb_to_hex(int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t))

    def _animate_to(self, target, steps=8):
        start = self._anim_color
        def step(i=0):
            if i >= steps:
                self._anim_color = target
                self._draw(target)
                return
            col = self._lerp_color(start, target, i / steps)
            self._anim_color = col
            self._draw(col)
            self.after(12, lambda: step(i+1))
        step()

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        self.create_arc(x1,    y1,    x1+2*r, y1+2*r, start=90,  extent=90,  **kw)
        self.create_arc(x2-2*r,y1,    x2,     y1+2*r, start=0,   extent=90,  **kw)
        self.create_arc(x1,    y2-2*r,x1+2*r, y2,     start=180, extent=90,  **kw)
        self.create_arc(x2-2*r,y2-2*r,x2,     y2,     start=270, extent=90,  **kw)
        self.create_rectangle(x1+r, y1,   x2-r, y2,   **kw)
        self.create_rectangle(x1,   y1+r, x2,   y2-r, **kw)

    def _draw(self, color):
        self.delete("all")
        self._rounded_rect(1, 1, self.w-1, self.h-1, self.radius, fill=color, outline="")
        self.create_text(self.w//2, self.h//2, text=self.label,
                         fill=self.fg_color, font=self.font, anchor="center")

    def _on_enter(self, _):  self._animate_to(self.hover_col)
    def _on_leave(self, _):  self._animate_to(self.bg_color)
    def _on_press(self, _):  self._draw(self.press_col)
    def _on_release(self, e):
        self._animate_to(self.hover_col)
        if self.command:
            self.after(80, self.command)


class StatusBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=CARD3, height=3, **kwargs)
        self.bar = tk.Frame(self, bg=ACCENT, width=0, height=3)
        self.bar.place(x=0, y=0, relheight=1)
        self._pulsing = False

    def set_progress(self, pct):
        self._pulsing = False
        self.update_idletasks()
        self.bar.config(width=int(self.winfo_width() * pct))

    def start_pulse(self, color=ACCENT):
        self._pulsing = True
        self.bar.config(bg=color)
        self._run_pulse()

    def _run_pulse(self, t=0):
        if not self._pulsing:
            return
        self.update_idletasks()
        total_w = max(self.winfo_width(), 1)
        x = int(((t % 60) / 60.0) * (total_w + 80)) - 80
        self.bar.place(x=x, y=0, relheight=1, width=80)
        self.after(16, lambda: self._run_pulse(t+1))

    def stop(self):
        self._pulsing = False
        self.bar.place(x=0, y=0, relheight=1, width=0)


class SectionHeader(tk.Frame):
    def __init__(self, parent, title, icon="", accent=ACCENT, **kwargs):
        bg = parent.cget("bg") if hasattr(parent, "cget") else CARD
        super().__init__(parent, bg=bg, **kwargs)
        tk.Frame(self, bg=accent, width=3).pack(side="left", fill="y", padx=(0, 10))
        tk.Label(self, text=f"{icon}  {title}" if icon else title,
                 fg=TEXT2, bg=bg, font=("Consolas", 8, "bold")).pack(side="left")


class ActivityCard(tk.Frame):
    def __init__(self, parent, entry, on_click, **kwargs):
        super().__init__(parent, bg=CARD2, padx=10, pady=8,
                         highlightbackground=BORDER, highlightthickness=1,
                         cursor="hand2", **kwargs)
        self._normal_bg = CARD2
        self._hover_bg  = CARD3
        self.bind("<Enter>", self._hover_in)
        self.bind("<Leave>", self._hover_out)

        try:
            clean_n = (entry['Name'].replace('✅ ', '').replace('⚠️ ', '')
                       .split(' already')[0].split(' Marked')[0])
            folder_name = f"{entry['NIC']} {clean_n}"
            img_path = os.path.join('images', folder_name)
            files = [f for f in os.listdir(img_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if files:
                pil_img = Image.open(os.path.join(img_path, files[0])).resize(
                    (44, 44), Image.Resampling.LANCZOS)
                tk_img = ImageTk.PhotoImage(pil_img)
                img_lbl = tk.Label(self, image=tk_img, bg=CARD2,
                                   highlightbackground=BORDER2, highlightthickness=1,
                                   cursor="hand2")
                img_lbl.image = tk_img
                img_lbl.pack(side="left", padx=(0, 10))
                img_lbl.bind("<Button-1>", lambda e: on_click())
            else:
                raise FileNotFoundError
        except:
            av = tk.Label(self, text="?", fg=MUTED, bg=CARD3,
                          font=("Consolas", 14, "bold"), width=3, height=2)
            av.pack(side="left", padx=(0, 10))

        info_f = tk.Frame(self, bg=CARD2, cursor="hand2")
        info_f.pack(side="left", fill="both", expand=True)

        name_text = entry['Name']
        name_color = ACCENT if "✅" in name_text else (ACCENT3 if "⚠️" in name_text else TEXT)
        tk.Label(info_f, text=name_text, fg=name_color, bg=CARD2,
                 font=("Segoe UI", 8, "bold"), anchor="w").pack(fill="x")
        tk.Label(info_f, text=entry['NIC'], fg=MUTED, bg=CARD2,
                 font=FONT_MONO_S, anchor="w").pack(fill="x")
        tk.Label(info_f, text=entry['Time'], fg=WARN, bg=CARD2,
                 font=FONT_TINY, anchor="w").pack(fill="x")

        for w in [self, info_f] + list(info_f.winfo_children()):
            try:
                w.bind("<Button-1>", lambda e: on_click())
                w.bind("<Enter>", self._hover_in)
                w.bind("<Leave>", self._hover_out)
            except:
                pass

    def _set_bg(self, color):
        self.config(bg=color)
        for w in self.winfo_children():
            try:
                w.config(bg=color)
                for ww in w.winfo_children():
                    try: ww.config(bg=color)
                    except: pass
            except:
                pass

    def _hover_in(self, _=None):  self._set_bg(self._hover_bg)
    def _hover_out(self, _=None): self._set_bg(self._normal_bg)


# ─────────────────────────────────────────────
#  REAL-TIME DASHBOARD  (hourly bar chart)
# ─────────────────────────────────────────────
_dash_fig   = None
_dash_ax    = None
_dash_canvas_widget = None

def build_dashboard_frame(parent):
    global _dash_fig, _dash_ax, _dash_canvas_widget

    plt.rcParams.update({
        "figure.facecolor": CARD,
        "axes.facecolor":   CARD2,
        "axes.edgecolor":   BORDER2,
        "axes.labelcolor":  MUTED,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "text.color":       TEXT2,
        "grid.color":       BORDER,
        "font.family":      "monospace",
        "font.size":        7,
    })

    _dash_fig, _dash_ax = plt.subplots(figsize=(4.6, 2.2))
    _dash_fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.22)

    _dash_canvas_widget = FigureCanvasTkAgg(_dash_fig, master=parent)
    _dash_canvas_widget.get_tk_widget().pack(fill="both", expand=True)

    _draw_empty_chart()


def _draw_empty_chart():
    _dash_ax.clear()
    hours = list(range(24))
    _dash_ax.bar(hours, [0]*24, color=MUTED2, width=0.65, zorder=2)
    _dash_ax.set_xlim(-0.6, 23.6)
    _dash_ax.set_ylim(0, 1)
    _dash_ax.set_xlabel("Hour of day", fontsize=6.5)
    _dash_ax.set_ylabel("Count", fontsize=6.5)
    _dash_ax.set_title("TODAY'S ATTENDANCE  ·  BY HOUR", fontsize=7,
                        color=ACCENT2, fontweight="bold", pad=4)
    _dash_ax.set_xticks(range(0, 24, 2))
    _dash_ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=5.5)
    _dash_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _dash_ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=1)
    _dash_ax.spines["top"].set_visible(False)
    _dash_ax.spines["right"].set_visible(False)
    _dash_fig.canvas.draw_idle()


def update_dashboard():
    def _do_update():
        log_file = "attendance_log.csv"
        today = datetime.now().strftime("%Y-%m-%d")
        counts = [0] * 24

        if os.path.exists(log_file):
            try:
                df = pd.read_csv(log_file, names=["NIC", "Name", "DateTime"],
                                 header=None, on_bad_lines="skip", engine="python")
                df["CleanDT"] = df["DateTime"].str.replace(" | ", " ", regex=False)
                df["DT"] = pd.to_datetime(df["CleanDT"], errors="coerce")
                df = df.dropna(subset=["DT"])
                df["Date"] = df["DT"].dt.strftime("%Y-%m-%d")
                df["Hour"] = df["DT"].dt.hour
                today_df = df[df["Date"] == today]
                hourly = today_df.groupby("Hour").size()
                for h, c in hourly.items():
                    if 0 <= h < 24:
                        counts[h] = int(c)
            except Exception:
                pass

        _dash_ax.clear()
        hours = list(range(24))
        max_val = max(counts) if max(counts) > 0 else 1
        bar_colors = [
            f"#{int(40 + 215 * (c / max_val)):02x}"
            f"{int(40 + 215 * (c / max_val)):02x}"
            f"{int(40 + 215 * (c / max_val)):02x}"
            for c in counts
        ]
        _dash_ax.bar(hours, counts, color=bar_colors, width=0.65, zorder=2)
        _dash_ax.set_xlim(-0.6, 23.6)
        _dash_ax.set_ylim(0, max(max_val + 1, 2))
        _dash_ax.set_xlabel("Hour of day", fontsize=6.5)
        _dash_ax.set_ylabel("Count", fontsize=6.5)
        _dash_ax.set_title("TODAY'S ATTENDANCE  ·  BY HOUR", fontsize=7,
                            color=ACCENT2, fontweight="bold", pad=4)
        _dash_ax.set_xticks(range(0, 24, 2))
        _dash_ax.set_xticklabels([f"{h:02d}" for h in range(0, 24, 2)], fontsize=5.5)
        _dash_ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _dash_ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=1)
        _dash_ax.spines["top"].set_visible(False)
        _dash_ax.spines["right"].set_visible(False)

        total = sum(counts)
        _dash_ax.text(0.97, 0.94, f"Total: {total}",
                      transform=_dash_ax.transAxes,
                      ha="right", va="top", fontsize=6.5,
                      color=ACCENT, fontweight="bold")

        _dash_fig.canvas.draw_idle()
        _dash_canvas_widget.get_tk_widget().update()

    root.after(0, _do_update)


# ─────────────────────────────────────────────
#  STUDENT DETAIL PANEL
# ─────────────────────────────────────────────
current_detail_student = None

def show_student_detail(nic, name, time_str, animate=True):
    global current_detail_student
    current_detail_student = {'NIC': nic, 'Name': name, 'Time': time_str}

    for widget in detail_content_frame.winfo_children():
        widget.destroy()

    clean_name = (name.replace("✅ ", "").replace("⚠️ ", "")
                  .split(" already")[0].split(" Marked")[0])
    db = get_student_db()
    student_info = db.get(clean_name.upper().strip(), {})

    hero = tk.Frame(detail_content_frame, bg=CARD2,
                    highlightbackground=BORDER2, highlightthickness=1)
    hero.pack(fill="x", padx=10, pady=(10, 6))
    tk.Frame(hero, bg=ACCENT, height=3).pack(fill="x")

    try:
        folder_name = f"{nic} {clean_name}"
        img_path = os.path.join('images', folder_name)
        files = [f for f in os.listdir(img_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            pil_img = Image.open(os.path.join(img_path, files[0])).resize(
                (80, 80), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl_photo = tk.Label(hero, image=tk_img, bg=CARD2,
                                 highlightbackground=ACCENT, highlightthickness=2)
            lbl_photo.image = tk_img
            lbl_photo.pack(pady=(12, 4))
        else:
            tk.Label(hero, text="👤", fg=MUTED, bg=CARD2, font=("Arial", 30)).pack(pady=(12,4))
    except:
        tk.Label(hero, text="👤", fg=MUTED, bg=CARD2, font=("Arial", 30)).pack(pady=(12,4))

    tk.Label(hero, text=clean_name, fg=ACCENT, bg=CARD2,
             font=("Consolas", 11, "bold")).pack()
    tk.Label(hero, text=f"NIC: {nic}", fg=MUTED, bg=CARD2, font=FONT_MONO_S).pack(pady=(2, 10))

    details = [
        ("⏰", "Marked At",  time_str),
        ("📚", "Grade",      student_info.get("Grade", "—")),
        ("🏠", "Address",    student_info.get("Address", "—")),
        ("📱", "Mobile",     str(student_info.get("Personal_Number", "—"))),
        ("☎️",  "Home",       str(student_info.get("Home_Number", "—"))),
    ]
    for icon, lbl, val in details:
        row = tk.Frame(detail_content_frame, bg=CARD,
                       highlightbackground=BORDER, highlightthickness=1)
        row.pack(fill="x", padx=10, pady=1)
        tk.Label(row, text=icon, bg=CARD, font=("Segoe UI", 9), width=2).pack(
            side="left", padx=(8,0), pady=6)
        tk.Label(row, text=lbl, fg=MUTED, bg=CARD, font=FONT_MONO_S,
                 width=9, anchor="w").pack(side="left", padx=4)
        tk.Label(row, text=val, fg=TEXT, bg=CARD, font=FONT_BODY,
                 anchor="w", wraplength=145).pack(side="left", padx=4, fill="x", expand=True)

    count = get_today_attendance_count(nic)
    count_frame = tk.Frame(detail_content_frame, bg=BG2,
                           highlightbackground=ACCENT, highlightthickness=1)
    count_frame.pack(fill="x", padx=10, pady=(8, 4))
    inner_count = tk.Frame(count_frame, bg=BG2)
    inner_count.pack(pady=10)
    tk.Label(inner_count, text="TODAY'S SCANS", fg=MUTED, bg=BG2,
             font=("Consolas", 7, "bold")).pack()
    tk.Label(inner_count, text=str(count), fg=ACCENT, bg=BG2,
             font=("Consolas", 32, "bold")).pack()
    tk.Label(inner_count, text="attendances recorded", fg=MUTED, bg=BG2,
             font=FONT_TINY).pack()

    try:
        folder_name = f"{nic} {clean_name}"
        img_path = os.path.join('images', folder_name)
        files = [f for f in os.listdir(img_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > 1:
            tk.Label(detail_content_frame, text="REGISTERED PHOTOS",
                     fg=MUTED, bg=CARD, font=("Consolas", 7, "bold")).pack(
                         anchor="w", padx=12, pady=(8,2))
            thumb_row = tk.Frame(detail_content_frame, bg=CARD)
            thumb_row.pack(fill="x", padx=10)
            for fname in files[:4]:
                try:
                    pi = Image.open(os.path.join(img_path, fname)).resize(
                        (44, 44), Image.Resampling.LANCZOS)
                    ti = ImageTk.PhotoImage(pi)
                    lbl = tk.Label(thumb_row, image=ti, bg=CARD,
                                   highlightbackground=BORDER2, highlightthickness=1)
                    lbl.image = ti
                    lbl.pack(side="left", padx=3, pady=4)
                except:
                    pass
    except:
        pass

    if animate:
        pulse_widget(hero, BORDER2, ACCENT, times=2, interval=150)


def get_today_attendance_count(nic):
    log_file = 'attendance_log.csv'
    if not os.path.exists(log_file):
        return 0
    try:
        df = pd.read_csv(log_file, names=['NIC', 'Name', 'DateTime'],
                         header=None, on_bad_lines='skip', engine='python')
        today = datetime.now().strftime('%Y-%m-%d')
        df['Date'] = df['DateTime'].str[:10]
        return len(df[(df['NIC'].astype(str) == str(nic)) & (df['Date'] == today)])
    except:
        return 0


# ─────────────────────────────────────────────
#  RECENT ACTIVITY SIDEBAR
# ─────────────────────────────────────────────
def update_recent_sidebar():
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    if not recent_logs:
        tk.Label(scrollable_frame, text="No recent activity",
                 fg=MUTED, bg=CARD, font=FONT_SMALL).pack(pady=20)
        return

    for entry in recent_logs[-6:][::-1]:
        def make_cb(e=entry):
            clean = (e['Name'].replace("✅ ", "").replace("⚠️ ", "")
                     .split(" already")[0].split(" Marked")[0])
            return lambda: show_student_detail(e['NIC'], clean, e['Time'])
        card = ActivityCard(scrollable_frame, entry, on_click=make_cb())
        card.pack(fill="x", pady=3, padx=4)

    canvas_mid.yview_moveto(0)


# ─────────────────────────────────────────────
#  ATTENDANCE MARKING  (CLAHE + voice feedback)
# ─────────────────────────────────────────────
def mark_attendance():
    global recent_logs
    db = get_student_db()
    model = "trained_face_model.pkl"
    if not os.path.exists(model):
        show_toast("⚠ Please train system first!", ACCENT3)
        return
    with open(model, "rb") as f:
        data = pickle.load(f)
    kEnc, kNames = data["encodings"], data["names"]

    status_bar.start_pulse(ACCENT)
    animate_label_fade(lbl_status, "● Scanner active — looking for faces…", INFO)

    cap = cv2.VideoCapture(0)
    found = False
    while True:
        ret, raw_img = cap.read()
        if not ret:
            break

        img = apply_clahe(raw_img)

        imgS = cv2.cvtColor(cv2.resize(img, (0, 0), None, 0.25, 0.25), cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(imgS)
        encs  = face_recognition.face_encodings(imgS, boxes)

        for enc, box in zip(encs, boxes):
            dis = face_recognition.face_distance(kEnc, enc)
            idx = np.argmin(dis)
            y1, x2, y2, x1 = [v * 4 for v in box]

            if dis[idx] < 0.45:
                name_folder = kNames[idx].upper().strip()
                for s_name in db:
                    if s_name in name_folder:
                        det          = db[s_name]
                        student_name = det['Name']
                        now_time     = datetime.now().strftime('%H:%M:%S')
                        cur_date     = datetime.now().strftime('%Y-%m-%d')

                        if is_already_marked_today(det['NIC'], cur_date):
                            animate_label_fade(lbl_status,
                                               f"⚠  {student_name} — already recorded", WARN)
                            show_toast(f"⚠ {student_name} already marked!", WARN)
                            speak(f"{student_name}, you are already marked")
                            log_entry = {'NIC': str(det['NIC']),
                                         'Name': f"⚠️ {student_name}",
                                         'Time': now_time}
                            recent_logs.append(log_entry)
                            update_recent_sidebar()
                            show_student_detail(str(det['NIC']), student_name, now_time)
                            cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 2)
                            cv2.putText(img, "Already marked", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
                        else:
                            animate_label_fade(lbl_status,
                                               f"✓  {student_name} marked successfully", ACCENT)
                            show_toast(f"✓ {student_name} marked!", ACCENT)
                            speak(f"Attendance Marked. Welcome {student_name}")
                            with open('attendance_log.csv', 'a') as lf:
                                lf.write(f"{det['NIC']},{student_name},"
                                         f"{datetime.now().strftime('%Y-%m-%d | %H:%M:%S')}\n")
                            log_entry = {'NIC': str(det['NIC']),
                                         'Name': f"✅ {student_name} Marked Successfully",
                                         'Time': now_time}
                            recent_logs.append(log_entry)
                            update_recent_sidebar()
                            show_student_detail(str(det['NIC']), student_name, now_time)
                            update_dashboard()
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            cv2.putText(img, student_name, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        found = True
                        break
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 2)
                animate_label_fade(lbl_status, "⚠  Unknown face detected", ACCENT3)

        cv2.imshow("SAP  ·  Scanner  |  Press Q to stop", img)
        if found or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    status_bar.stop()
    if not found:
        animate_label_fade(lbl_status, "● Scanner closed — no match found", MUTED)


def is_already_marked_today(nic, current_date):
    log_file = 'attendance_log.csv'
    if not os.path.exists(log_file):
        return False
    try:
        df = pd.read_csv(log_file, names=['NIC', 'Name', 'DateTime'],
                         header=None, on_bad_lines='skip', engine='python')
        df['Date'] = df['DateTime'].str[:10]
        return len(df[(df['NIC'].astype(str) == str(nic)) & (df['Date'] == current_date)]) > 0
    except:
        return False


# ─────────────────────────────────────────────
#  STUDENT DB
# ─────────────────────────────────────────────
def get_student_db():
    try:
        csv_path = 'students.csv' if os.path.exists('students.csv') else resource_path('students.csv')
        df = pd.read_csv(csv_path)
        return df.set_index(df['Name'].str.upper().str.strip()).to_dict('index')
    except:
        return {}


# ─────────────────────────────────────────────
#  ADMIN MANAGEMENT WINDOW  (with real-time search)
# ─────────────────────────────────────────────
def open_admin_panel():
    win = tk.Toplevel(root)
    win.title("Admin Management — SAP")
    win.geometry("860x600")
    win.configure(bg=BG)
    win.resizable(True, True)

    win.update_idletasks()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"860x600+{(sw-860)//2}+{(sh-600)//2}")

    # ── Header ──────────────────────────────────
    hdr = tk.Frame(win, bg=CARD2, height=54)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Frame(hdr, bg=ACCENT2, width=4).pack(side="left", fill="y")
    tk.Label(hdr, text="  STUDENT MANAGEMENT",
             fg=TEXT, bg=CARD2, font=("Consolas", 13, "bold")).pack(side="left", padx=16)
    lbl_count = tk.Label(hdr, text="", fg=MUTED, bg=CARD2, font=FONT_MONO_S)
    lbl_count.pack(side="right", padx=16)

    tk.Frame(win, bg=BORDER2, height=1).pack(fill="x")

    # ══════════════════════════════════════════════
    #  SEARCH BAR  –  real-time, KeyRelease-driven
    # ══════════════════════════════════════════════
    search_outer = tk.Frame(win, bg=BG2, pady=10)
    search_outer.pack(fill="x", padx=16, pady=(10, 0))

    tk.Label(search_outer,
             text="🔍  Search by Name or NIC",
             fg=MUTED, bg=BG2,
             font=("Consolas", 8, "bold")).pack(anchor="w", pady=(0, 4))

    # Outer border frame — colour changes on focus
    search_border = tk.Frame(search_outer, bg=BORDER2,
                             highlightbackground=BORDER2,
                             highlightthickness=1)
    search_border.pack(fill="x")

    # Inner background frame (1-px inset from border)
    search_inner = tk.Frame(search_border, bg=CARD2)
    search_inner.pack(fill="both", padx=1, pady=1)

    # Magnifier icon label (dims / brightens with focus)
    lbl_search_icon = tk.Label(search_inner, text="◈", fg=MUTED, bg=CARD2,
                               font=("Consolas", 10), width=2)
    lbl_search_icon.pack(side="left", padx=(8, 0))

    PLACEHOLDER = "Type to filter…"

    search_entry = tk.Entry(
        search_inner,
        bg=CARD2, fg=MUTED,
        insertbackground=ACCENT,
        relief="flat", font=FONT_BODY, bd=0,
    )
    search_entry.insert(0, PLACEHOLDER)
    search_entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(6, 8))

    # Thin separator under the text area
    tk.Frame(search_border, bg=BORDER, height=1).pack(fill="x")

    # ── Focus: glow border + clear placeholder ──
    def _search_focus_in(event=None):
        search_border.config(highlightbackground=ACCENT)
        lbl_search_icon.config(fg=ACCENT)
        if search_entry.get() == PLACEHOLDER:
            search_entry.delete(0, "end")
            search_entry.config(fg=TEXT)

    def _search_focus_out(event=None):
        search_border.config(highlightbackground=BORDER2)
        lbl_search_icon.config(fg=MUTED)
        if search_entry.get().strip() == "":
            search_entry.insert(0, PLACEHOLDER)
            search_entry.config(fg=MUTED)

    search_entry.bind("<FocusIn>",  _search_focus_in)
    search_entry.bind("<FocusOut>", _search_focus_out)

    # ── Treeview ────────────────────────────────
    tree_frame = tk.Frame(win, bg=CARD)
    tree_frame.pack(fill="both", expand=True, padx=16, pady=10)

    cols = ("NIC", "Name", "Grade", "Personal_Number", "Home_Number", "Address")
    col_widths = (110, 160, 80, 110, 110, 200)

    tree_style = ttk.Style()
    tree_style.configure("Admin.Treeview",
                         background=CARD,
                         foreground=TEXT,
                         fieldbackground=CARD,
                         rowheight=28,
                         font=FONT_BODY)
    tree_style.configure("Admin.Treeview.Heading",
                         background=CARD3,
                         foreground=ACCENT2,
                         font=("Consolas", 8, "bold"),
                         relief="flat")
    tree_style.map("Admin.Treeview",
                   background=[("selected", CARD3)],
                   foreground=[("selected", ACCENT)])

    vsb = ttk.Scrollbar(tree_frame, orient="vertical")
    hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

    tree = ttk.Treeview(tree_frame, columns=cols, show="headings",
                        style="Admin.Treeview",
                        yscrollcommand=vsb.set,
                        xscrollcommand=hsb.set)

    for col, w in zip(cols, col_widths):
        tree.heading(col, text=col.upper().replace("_", " "))
        tree.column(col, width=w, minwidth=60, anchor="w")

    vsb.config(command=tree.yview)
    hsb.config(command=tree.xview)

    vsb.pack(side="right", fill="y")
    hsb.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    tree.tag_configure("odd",  background=CARD2)
    tree.tag_configure("even", background=CARD)

    # ══════════════════════════════════════════════
    #  FILTER LOGIC
    # ══════════════════════════════════════════════
    def _populate_tree(df):
        """Wipe the Treeview and refill it from *df*."""
        for item in tree.get_children():
            tree.delete(item)
        for i, (_, row) in enumerate(df.iterrows()):
            tag = "odd" if i % 2 == 0 else "even"
            tree.insert("", "end",
                        iid=str(row.get("NIC", i)),
                        values=(
                            str(row.get("NIC", "")),
                            str(row.get("Name", "")),
                            str(row.get("Grade", "")),
                            str(row.get("Personal_Number", "")),
                            str(row.get("Home_Number", "")),
                            str(row.get("Address", "")),
                        ),
                        tags=(tag,))

    def load_students(filter_text=""):
        """
        Read students.csv, apply an optional filter across Name + NIC
        columns, repopulate the Treeview, and update the count label.

        When *filter_text* is empty the full dataset is shown.
        """
        csv_path = "students.csv"
        if not os.path.exists(csv_path):
            lbl_count.config(text="No students.csv found")
            return

        try:
            df = pd.read_csv(csv_path)

            # Guarantee every expected column is present
            for c in ("NIC", "Name", "Grade", "Personal_Number", "Home_Number", "Address"):
                if c not in df.columns:
                    df[c] = ""

            if filter_text:
                # Vectorised, case-insensitive OR across Name and NIC
                q = filter_text.lower()
                mask = (
                    df["Name"].astype(str).str.lower().str.contains(q, na=False)
                    | df["NIC"].astype(str).str.lower().str.contains(q, na=False)
                )
                df = df[mask]

            _populate_tree(df)

            total = len(df)
            suffix = "student" if total == 1 else "students"
            label = f"{total} {suffix} loaded"
            if filter_text:
                label += f'  (filter: "{filter_text}")'
            lbl_count.config(text=label)

        except Exception as e:
            lbl_count.config(text=f"Error: {e}")

    def filter_students(event=None):
        """
        Bound to <KeyRelease> on the search entry.

        Reads the current entry text, ignores the placeholder string,
        and calls load_students() so the Treeview updates instantly.
        """
        raw = search_entry.get().strip()
        # Treat the greyed-out placeholder as "no filter"
        query = "" if raw == PLACEHOLDER else raw
        load_students(query)

    # Every key release triggers an instant re-filter
    search_entry.bind("<KeyRelease>", filter_students)

    # Initial population — no filter
    load_students()

    # ── Action buttons ───────────────────────────
    btn_bar = tk.Frame(win, bg=BG, pady=10)
    btn_bar.pack(fill="x", padx=16)

    # ── Edit ────────────────────────────────────
    def edit_student():
        sel = tree.selection()
        if not sel:
            show_toast("Select a student to edit", WARN)
            return

        values = tree.item(sel[0], "values")
        nic, name, grade, personal, home, address = values

        edit_win = tk.Toplevel(win)
        edit_win.title(f"Edit — {name}")
        edit_win.geometry("480x400")
        edit_win.configure(bg=BG)
        edit_win.resizable(False, False)
        edit_win.grab_set()

        edit_win.update_idletasks()
        sw2, sh2 = edit_win.winfo_screenwidth(), edit_win.winfo_screenheight()
        edit_win.geometry(f"480x400+{(sw2-480)//2}+{(sh2-400)//2}")

        tk.Frame(edit_win, bg=ACCENT, height=3).pack(fill="x")
        tk.Label(edit_win, text=f"  EDITING: {name}",
                 fg=ACCENT, bg=BG, font=("Consolas", 10, "bold")).pack(
                     anchor="w", padx=16, pady=(12, 6))

        fields_data = [
            ("Name",            name),
            ("Grade",           grade),
            ("Personal_Number", personal),
            ("Home_Number",     home),
            ("Address",         address),
        ]
        entries = {}
        for label, default in fields_data:
            rf = tk.Frame(edit_win, bg=BG)
            rf.pack(fill="x", padx=16, pady=3)
            tk.Label(rf, text=label.replace("_", " "), fg=MUTED, bg=BG,
                     font=FONT_TINY, width=16, anchor="w").pack(side="left")
            e = tk.Entry(rf, bg=CARD2, fg=TEXT, insertbackground=ACCENT,
                         relief="flat", font=FONT_BODY, bd=0)
            e.insert(0, default)
            e.pack(side="left", fill="x", expand=True, ipady=5, padx=(4, 0))
            tk.Frame(rf, bg=BORDER, height=1).pack(side="bottom", fill="x")
            entries[label] = e

        def save_edit():
            csv_path = "students.csv"
            if not os.path.exists(csv_path):
                show_toast("students.csv not found", ACCENT3)
                return
            try:
                df = pd.read_csv(csv_path)
                idx = df[df["NIC"].astype(str) == str(nic)].index
                if idx.empty:
                    show_toast("Student not found in CSV", ACCENT3)
                    return
                old_name = df.loc[idx[0], "Name"]
                for label, entry in entries.items():
                    df.loc[idx[0], label] = entry.get().strip()
                new_name = df.loc[idx[0], "Name"]
                df.to_csv(csv_path, index=False)

                # Rename the image folder if the student's name changed
                if old_name.strip() != new_name.strip():
                    old_folder = os.path.join("images", f"{nic} {old_name.strip()}")
                    new_folder = os.path.join("images", f"{nic} {new_name.strip()}")
                    if os.path.exists(old_folder):
                        os.rename(old_folder, new_folder)

                show_toast(f"✓ {new_name} updated!", ACCENT)
                edit_win.destroy()
                # Re-apply the active search filter after saving
                filter_students()
            except Exception as e:
                show_toast(f"Error: {e}", ACCENT3)

        GlowButton(edit_win, "SAVE CHANGES", command=save_edit,
                   bg_color=ACCENT, fg_color=BG, width=444, height=42,
                   icon="✅", font=("Consolas", 10, "bold")).pack(padx=16, pady=(16, 8))

    # ── Delete ───────────────────────────────────
    def delete_student():
        sel = tree.selection()
        if not sel:
            show_toast("Select a student to delete", WARN)
            return

        values = tree.item(sel[0], "values")
        nic, name = values[0], values[1]

        confirmed = messagebox.askyesno(
            "Confirm Delete",
            f"Permanently delete {name} (NIC: {nic})?\n\n"
            "This will remove their CSV record AND image folder.",
            icon="warning",
            parent=win,
        )
        if not confirmed:
            return

        errors = []
        csv_path = "students.csv"
        try:
            df = pd.read_csv(csv_path)
            df = df[df["NIC"].astype(str) != str(nic)]
            df.to_csv(csv_path, index=False)
        except Exception as e:
            errors.append(f"CSV: {e}")

        folder = os.path.join("images", f"{nic} {name}")
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
            except Exception as e:
                errors.append(f"Folder: {e}")

        if errors:
            show_toast(f"Partial delete: {'; '.join(errors)}", ACCENT3)
        else:
            show_toast(f"✓ {name} deleted", ACCENT)

        # Re-apply the active search filter after deleting
        filter_students()

    # ── Refresh ──────────────────────────────────
    def refresh_table():
        """Clear the search box and reload all students."""
        search_entry.delete(0, "end")
        search_entry.insert(0, PLACEHOLDER)
        search_entry.config(fg=MUTED)
        search_border.config(highlightbackground=BORDER2)
        lbl_search_icon.config(fg=MUTED)
        load_students()
        show_toast("Table refreshed", INFO)

    GlowButton(btn_bar, "Edit Selected",   command=edit_student,
               bg_color=CARD3, fg_color=ACCENT,  width=180, height=38, icon="✏").pack(
                   side="left", padx=(0, 8))
    GlowButton(btn_bar, "Delete Selected", command=delete_student,
               bg_color=CARD3, fg_color=ACCENT3, width=180, height=38, icon="🗑").pack(
                   side="left", padx=(0, 8))
    GlowButton(btn_bar, "Refresh",         command=refresh_table,
               bg_color=CARD3, fg_color=MUTED,   width=120, height=38, icon="↻").pack(
                   side="left")

    tk.Label(btn_bar, text="Double-click a row to edit",
             fg=MUTED, bg=BG, font=FONT_TINY).pack(side="right", padx=4)

    # Double-click shortcut to edit
    tree.bind("<Double-1>", lambda e: edit_student())


# ─────────────────────────────────────────────
#  ADD STUDENT WINDOW
# ─────────────────────────────────────────────
def open_add_student_window():
    win = tk.Toplevel(root)
    win.title("Register Student — SAP")
    win.geometry("680x640")
    win.configure(bg=BG)
    win.resizable(False, False)

    win.update_idletasks()
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"680x640+{(sw-680)//2}+{(sh-640)//2}")

    hdr = tk.Frame(win, bg=CARD2, height=58)
    hdr.pack(fill="x")
    hdr.pack_propagate(False)
    tk.Frame(hdr, bg=ACCENT, width=4).pack(side="left", fill="y")
    tk.Label(hdr, text="  REGISTER NEW STUDENT",
             fg=TEXT, bg=CARD2, font=("Consolas", 13, "bold")).pack(side="left", padx=16)
    tk.Label(hdr, text="Fill all required fields (*)",
             fg=MUTED, bg=CARD2, font=FONT_SMALL).pack(side="right", padx=16)

    body_canvas = tk.Canvas(win, bg=BG, highlightthickness=0)
    sb = ttk.Scrollbar(win, orient="vertical", command=body_canvas.yview)
    sbody = tk.Frame(body_canvas, bg=BG)
    sbody.bind("<Configure>",
               lambda e: body_canvas.configure(scrollregion=body_canvas.bbox("all")))
    body_canvas.create_window((0, 0), window=sbody, anchor="nw", width=680)
    body_canvas.configure(yscrollcommand=sb.set)
    body_canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def make_section(parent, title, icon, accent=ACCENT):
        sf = tk.Frame(parent, bg=CARD, highlightbackground=BORDER2, highlightthickness=1)
        sf.pack(fill="x", padx=16, pady=(14, 0))
        hf = tk.Frame(sf, bg=CARD)
        hf.pack(fill="x", padx=14, pady=(10, 6))
        tk.Frame(hf, bg=accent, width=3).pack(side="left", fill="y")
        tk.Label(hf, text=f"  {icon}  {title}", fg=TEXT2, bg=CARD,
                 font=("Consolas", 9, "bold")).pack(side="left")
        tk.Frame(sf, bg=BORDER, height=1).pack(fill="x", padx=12)
        return sf

    def make_field(parent, label, required=False):
        ff = tk.Frame(parent, bg=CARD2, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(ff, text=f"{label} *" if required else label,
                 fg=MUTED, bg=CARD2, font=FONT_TINY).pack(anchor="w", padx=10, pady=(7, 0))
        e = tk.Entry(ff, bg=CARD2, fg=TEXT, insertbackground=ACCENT,
                     relief="flat", font=FONT_BODY, bd=0)
        e.pack(fill="x", padx=10, pady=(2, 0), ipady=5)
        tk.Frame(ff, bg=BORDER, height=1).pack(fill="x", padx=10, pady=(0, 6))
        e.bind("<FocusIn>",  lambda _: ff.config(highlightbackground=ACCENT))
        e.bind("<FocusOut>", lambda _: ff.config(highlightbackground=BORDER))
        return ff, e

    pc = make_section(sbody, "PERSONAL INFORMATION", "📋")
    row1 = tk.Frame(pc, bg=CARD)
    row1.pack(fill="x", padx=14, pady=8)
    nf, nic_entry   = make_field(row1, "NIC Number", required=True)
    nf.pack(side="left", fill="both", expand=True, padx=(0, 6))
    nmf, name_entry = make_field(row1, "Full Name", required=True)
    nmf.pack(side="left", fill="both", expand=True, padx=(6, 0))

    gf_wrap = tk.Frame(pc, bg=CARD)
    gf_wrap.pack(fill="x", padx=14, pady=(0, 12))
    gff, grade_entry = make_field(gf_wrap, "Grade / Class")
    gff.pack(fill="x")

    cc = make_section(sbody, "CONTACT INFORMATION", "📞", ACCENT2)
    row3 = tk.Frame(cc, bg=CARD)
    row3.pack(fill="x", padx=14, pady=8)
    pf, personal_entry = make_field(row3, "Personal Phone")
    pf.pack(side="left", fill="both", expand=True, padx=(0, 6))
    hf2, home_entry    = make_field(row3, "Home Phone")
    hf2.pack(side="left", fill="both", expand=True, padx=(6, 0))

    af_wrap = tk.Frame(cc, bg=CARD)
    af_wrap.pack(fill="x", padx=14, pady=(0, 12))
    aff, address_entry = make_field(af_wrap, "Address")
    aff.pack(fill="x")

    phc = make_section(sbody, "STUDENT PHOTOS", "📸", WARN)
    selected_images = []

    ph_top = tk.Frame(phc, bg=CARD)
    ph_top.pack(fill="x", padx=14, pady=8)
    lbl_photo_status = tk.Label(ph_top, text="No images selected",
                                fg=MUTED, bg=CARD, font=FONT_BODY)
    lbl_photo_status.pack(side="left")

    photo_preview_frame = tk.Frame(phc, bg=CARD2, height=76,
                                   highlightbackground=BORDER, highlightthickness=1)
    photo_preview_frame.pack(fill="x", padx=14, pady=(0, 12))
    photo_preview_frame.pack_propagate(False)
    tk.Label(photo_preview_frame, text="Photos will appear here",
             fg=MUTED, bg=CARD2, font=FONT_SMALL).pack(pady=28)

    def update_preview():
        for w in photo_preview_frame.winfo_children():
            w.destroy()
        if selected_images:
            for p in selected_images[:5]:
                try:
                    pi = Image.open(p).resize((58, 58), Image.Resampling.LANCZOS)
                    ti = ImageTk.PhotoImage(pi)
                    lb = tk.Label(photo_preview_frame, image=ti, bg=CARD2,
                                  highlightbackground=BORDER2, highlightthickness=1)
                    lb.image = ti
                    lb.pack(side="left", padx=4, pady=8)
                except:
                    pass
            if len(selected_images) > 5:
                tk.Label(photo_preview_frame, text=f"+{len(selected_images)-5}",
                         fg=MUTED, bg=CARD2, font=("Consolas", 10, "bold")).pack(
                             side="left", padx=8, pady=8)
        else:
            tk.Label(photo_preview_frame, text="Photos will appear here",
                     fg=MUTED, bg=CARD2, font=FONT_SMALL).pack(pady=28)

    def select_images():
        nonlocal selected_images
        files = filedialog.askopenfilenames(title="Select Student Photos",
                                            filetypes=[("Images", "*.jpg *.jpeg *.png")])
        selected_images = list(files)
        lbl_photo_status.config(
            text=f"✔  {len(selected_images)} photo(s) selected" if selected_images else "No images selected",
            fg=ACCENT if selected_images else MUTED)
        update_preview()

    GlowButton(ph_top, "Browse Photos", command=select_images,
               bg_color=WARN, fg_color=BG, width=150, height=34, icon="📸").pack(side="right")

    sbtn_frame = tk.Frame(sbody, bg=BG)
    sbtn_frame.pack(fill="x", padx=16, pady=(16, 4))

    def submit():
        d = {
            'NIC':            nic_entry.get().strip(),
            'Name':           name_entry.get().strip(),
            'Grade':          grade_entry.get().strip(),
            'Address':        address_entry.get().strip(),
            'Personal_Number':personal_entry.get().strip(),
            'Home_Number':    home_entry.get().strip()
        }
        if not d['NIC'] or not d['Name'] or not selected_images:
            show_toast("⚠ NIC, Name & Photos required", ACCENT3)
            return
        try:
            csv_path = 'students.csv'
            nr = pd.DataFrame([d])
            if os.path.exists(csv_path):
                nr.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                nr.to_csv(csv_path, index=False)
            folder = os.path.join('images', f"{d['NIC']} {d['Name']}")
            os.makedirs(folder, exist_ok=True)
            for i, p in enumerate(selected_images):
                shutil.copy(p, os.path.join(folder, f"img_{i}.jpg"))
            show_toast(f"✓ {d['Name']} added! Retrain now.", ACCENT)
            win.destroy()
        except Exception as e:
            show_toast(f"Error: {e}", ACCENT3)

    GlowButton(sbtn_frame, "SAVE STUDENT", command=submit,
               bg_color=ACCENT, fg_color=BG, width=648, height=46,
               icon="✅", font=("Consolas", 12, "bold")).pack()

    tk.Label(sbody, text="* Required fields",
             fg=MUTED, bg=BG, font=FONT_TINY).pack(pady=(4, 16))


# ─────────────────────────────────────────────
#  DOWNLOAD REPORT
# ─────────────────────────────────────────────
def download_report():
    selected_date = cal.get_date().strftime('%Y-%m-%d')
    log_file = 'attendance_log.csv'
    if not os.path.exists(log_file):
        show_toast("No attendance records found", ACCENT3)
        return
    try:
        df = pd.read_csv(log_file, names=['NIC', 'Name', 'DateTime'],
                         header=None, on_bad_lines='skip', engine='python')
        df['CleanDT'] = df['DateTime'].str.replace(' | ', ' ', regex=False)
        df['DateOnly'] = pd.to_datetime(df['CleanDT'], errors='coerce').dt.strftime('%Y-%m-%d')
        filtered = df[df['DateOnly'] == selected_date]
        if filtered.empty:
            show_toast(f"No records for {selected_date}", WARN)
            return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                            filetypes=[("Excel", "*.xlsx")])
        if path:
            filtered[['NIC', 'Name', 'DateTime']].to_excel(path, index=False)
            show_toast("✓ Report exported successfully!", ACCENT)
    except Exception as e:
        show_toast(f"Export error: {e}", ACCENT3)


# ─────────────────────────────────────────────
#  TRAINING SYSTEM
# ─────────────────────────────────────────────
training_progress_log = []
is_training = False

def clear_detail_panel():
    for w in detail_content_frame.winfo_children():
        w.destroy()

def add_progress_message(message, icon="⚡"):
    global training_progress_log
    training_progress_log.append(f"{icon}  {message}")
    clear_detail_panel()

    tk.Label(detail_content_frame, text="TRAINING PROGRESS",
             fg=MUTED, bg=CARD, font=("Consolas", 8, "bold")).pack(
                 anchor="w", padx=12, pady=(10, 4))
    tk.Frame(detail_content_frame, bg=ACCENT, height=2).pack(fill="x", padx=10, pady=(0, 6))

    for msg in training_progress_log[-10:][::-1]:
        color = ACCENT if "✅" in msg else (ACCENT3 if "❌" in msg else (WARN if "⚠" in msg else TEXT2))
        mf = tk.Frame(detail_content_frame, bg=CARD2,
                      highlightbackground=BORDER, highlightthickness=1)
        mf.pack(fill="x", padx=10, pady=1)
        tk.Label(mf, text=msg, fg=color, bg=CARD2, font=FONT_MONO_S,
                 anchor="w", wraplength=270, justify="left").pack(anchor="w", padx=8, pady=4)

    root.update()


def show_training_complete(added, removed, total, failed, skipped):
    clear_detail_panel()

    hdr = tk.Frame(detail_content_frame, bg=BG2,
                   highlightbackground=ACCENT, highlightthickness=1)
    hdr.pack(fill="x", padx=10, pady=(10, 6))
    tk.Frame(hdr, bg=ACCENT, height=3).pack(fill="x")
    tk.Label(hdr, text="✅  TRAINING COMPLETE", fg=ACCENT, bg=BG2,
             font=("Consolas", 10, "bold")).pack(pady=(10, 6))

    for icon, lbl, val, col in [
        ("🆕", "Newly encoded",    str(added),   ACCENT),
        ("⏭",  "Already cached",   str(skipped), INFO),
        ("🗑",  "Removed",          str(removed), ACCENT3),
        ("❌",  "Failed (no face)", str(failed),  ACCENT3),
        ("📦",  "Total in model",   str(total),   ACCENT),
    ]:
        row = tk.Frame(detail_content_frame, bg=CARD2,
                       highlightbackground=BORDER, highlightthickness=1)
        row.pack(fill="x", padx=10, pady=2)
        tk.Label(row, text=icon, bg=CARD2, width=2).pack(side="left", padx=8, pady=6)
        tk.Label(row, text=lbl, fg=MUTED, bg=CARD2, font=FONT_SMALL,
                 width=16, anchor="w").pack(side="left")
        tk.Label(row, text=val, fg=col, bg=CARD2,
                 font=("Consolas", 10, "bold"), anchor="e").pack(side="right", padx=10)


def train_system_thread():
    global is_training, training_progress_log
    path = 'images'
    os.makedirs(path, exist_ok=True)
    model_file = "trained_face_model.pkl"

    add_progress_message("Initializing…", "🚀")
    add_progress_message("Scanning image directory…", "📂")

    disk_files = {}
    disk_folders = set()
    for root_dir, _, files in os.walk(path):
        folder_name = os.path.basename(root_dir)
        if folder_name != 'images':
            disk_folders.add(folder_name.upper().strip())
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                fp = os.path.join(root_dir, filename)
                try:
                    h = hashlib.md5(open(fp, 'rb').read()).hexdigest()
                    disk_files[fp] = h
                except:
                    pass

    disk_hashes = set(disk_files.values())
    add_progress_message(f"Found {len(disk_files)} images in {len(disk_folders)} folders", "📊")

    existing_enc, existing_names, existing_paths, existing_hashes = [], [], [], []
    if os.path.exists(model_file):
        add_progress_message("Loading existing model…", "📥")
        try:
            with open(model_file, "rb") as f:
                saved = pickle.load(f)
            existing_enc    = saved.get("encodings", [])
            existing_names  = saved.get("names",     [])
            existing_hashes = saved.get("hashes",    [])
            existing_paths  = saved.get("paths",     [""] * len(existing_enc))
            add_progress_message(f"Loaded {len(existing_enc)} encodings", "📦")
        except:
            add_progress_message("Starting fresh model", "✨")
    else:
        add_progress_message("Creating new model", "✨")

    add_progress_message("Pruning deleted images…", "🔍")
    keep_enc, keep_names, keep_paths, keep_hashes = [], [], [], []
    removed_count = 0
    for enc, name, fpath, fhash in zip(existing_enc, existing_names,
                                        existing_paths, existing_hashes):
        if fhash in disk_hashes:
            keep_enc.append(enc); keep_names.append(name)
            keep_paths.append(fpath); keep_hashes.append(fhash)
        else:
            removed_count += 1

    if removed_count:
        add_progress_message(f"Removed {removed_count} stale encodings", "🗑")

    trained_hashes_set = set(keep_hashes)
    new_enc, new_names, new_paths, new_hashes = [], [], [], []
    skipped = trained_new = failed = 0

    add_progress_message("Encoding new images…", "🧠")

    for f_path, h in disk_files.items():
        if h in trained_hashes_set:
            skipped += 1
            continue
        folder_name = os.path.basename(os.path.dirname(f_path))
        img_name    = os.path.basename(f_path)
        add_progress_message(f"{folder_name[:20]} — {img_name}", "⚡")
        try:
            img = cv2.imread(f_path)
            if img is None:
                failed += 1; add_progress_message(f"Read fail: {img_name}", "⚠"); continue
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            if encs:
                new_enc.append(encs[0])
                new_names.append(folder_name.upper().strip())
                new_paths.append(f_path)
                new_hashes.append(h)
                trained_new += 1
                add_progress_message(f"OK: {folder_name[:22]}", "✅")
            else:
                failed += 1
                add_progress_message(f"No face: {img_name}", "❌")
        except Exception:
            failed += 1
            add_progress_message(f"Error: {img_name}", "❌")

    add_progress_message("Saving model…", "💾")
    final_enc    = keep_enc    + new_enc
    final_names  = keep_names  + new_names
    final_paths  = keep_paths  + new_paths
    final_hashes = keep_hashes + new_hashes

    with open(model_file, "wb") as f:
        pickle.dump({"encodings": final_enc, "names": final_names,
                     "paths": final_paths,   "hashes": final_hashes}, f)

    is_training = False
    status_bar.stop()
    root.config(cursor="")
    show_training_complete(trained_new, removed_count, len(final_enc), failed, skipped)
    animate_label_fade(lbl_status, f"✓  Training done — {len(final_enc)} faces", ACCENT)
    show_toast(f"✓ Model trained: {len(final_enc)} encodings", ACCENT)


def start_training():
    global is_training, training_progress_log
    if is_training:
        show_toast("Training already in progress!", WARN)
        return
    training_progress_log = []
    is_training = True
    root.config(cursor="watch")
    clear_detail_panel()
    add_progress_message("Starting training…", "🚀")
    animate_label_fade(lbl_status, "🔄  Training in progress…", WARN)
    status_bar.start_pulse(WARN)
    threading.Thread(target=train_system_thread, daemon=True).start()


# ─────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────
root = tk.Tk()
root.title("Smart Attendance Pro  ·  v5.2")
root.geometry("1200x700")
root.configure(bg=BG)
root.resizable(False, False)

style = ttk.Style()
style.theme_use("clam")
style.configure("Vertical.TScrollbar", background=CARD3, troughcolor=CARD,
                bordercolor=BORDER, arrowcolor=MUTED, relief="flat", width=6)
style.configure("TScrollbar", background=CARD3, troughcolor=CARD,
                bordercolor=BORDER, arrowcolor=MUTED, relief="flat", width=6)

# ════════════════════════════════════════════════
#  LEFT PANEL
# ════════════════════════════════════════════════
left_panel = tk.Frame(root, bg=BG)
left_panel.pack(side="left", fill="both", expand=True, padx=(14, 8), pady=0)

# ── Top brand + clock ──
topbar = tk.Frame(left_panel, bg=BG)
topbar.pack(fill="x", pady=(14, 0))

brand_f = tk.Frame(topbar, bg=BG)
brand_f.pack(side="left")
tk.Label(brand_f, text="SAP", fg=ACCENT, bg=BG,
         font=("Consolas", 20, "bold")).pack(side="left")
tk.Label(brand_f, text="  Smart Attendance Pro", fg=MUTED, bg=BG,
         font=("Consolas", 9)).pack(side="left", pady=(4, 0))

clock_f = tk.Frame(topbar, bg=BG)
clock_f.pack(side="right")
lbl_time = tk.Label(clock_f, text="00:00:00", fg=ACCENT, bg=BG,
                    font=("Consolas", 20, "bold"))
lbl_time.pack(anchor="e")
lbl_date = tk.Label(clock_f, text="", fg=MUTED, bg=BG, font=FONT_MONO_S)
lbl_date.pack(anchor="e")

tk.Frame(left_panel, bg=BORDER2, height=1).pack(fill="x", pady=(10, 0))

# ── Scanner card ──
scanner_card = tk.Frame(left_panel, bg=CARD2,
                         highlightbackground=ACCENT, highlightthickness=1)
scanner_card.pack(fill="x", pady=(12, 0))

scan_inner = tk.Frame(scanner_card, bg=CARD2)
scan_inner.pack(pady=(14, 8), padx=16, fill="x")

GlowButton(scan_inner, "START  SCANNER", command=mark_attendance,
           bg_color=ACCENT, fg_color=BG, width=420, height=46,
           radius=12, icon="▶", font=("Consolas", 12, "bold")).pack()

lbl_status = tk.Label(scanner_card, text="● System ready",
                       fg=MUTED, bg=CARD2, font=("Segoe UI", 9, "italic"))
lbl_status.pack(pady=(0, 4))

status_bar = StatusBar(scanner_card)
status_bar.pack(fill="x")

# ── Dashboard chart ──
dash_card = tk.Frame(left_panel, bg=CARD,
                     highlightbackground=BORDER2, highlightthickness=1)
dash_card.pack(fill="x", pady=(10, 0))

dash_hdr = tk.Frame(dash_card, bg=CARD)
dash_hdr.pack(fill="x", padx=12, pady=(8, 0))
tk.Frame(dash_hdr, bg=WARN, width=3).pack(side="left", fill="y")
tk.Label(dash_hdr, text="  LIVE DASHBOARD", fg=TEXT2, bg=CARD,
         font=("Consolas", 8, "bold")).pack(side="left")

btn_refresh_dash = tk.Label(dash_hdr, text="↻ Refresh", fg=MUTED, bg=CARD,
                             font=FONT_TINY, cursor="hand2")
btn_refresh_dash.pack(side="right")
btn_refresh_dash.bind("<Button-1>", lambda e: update_dashboard())

tk.Frame(dash_card, bg=BORDER, height=1).pack(fill="x", padx=10, pady=(4, 0))

dash_inner = tk.Frame(dash_card, bg=CARD)
dash_inner.pack(fill="x", padx=6, pady=4)
build_dashboard_frame(dash_inner)

# ── Admin tools ──
admin_card = tk.Frame(left_panel, bg=CARD,
                       highlightbackground=BORDER2, highlightthickness=1)
admin_card.pack(fill="x", pady=(10, 0))

SectionHeader(admin_card, "ADMIN TOOLS", "⚙", ACCENT2).pack(
    anchor="w", padx=14, pady=(8, 4))
tk.Frame(admin_card, bg=BORDER, height=1).pack(fill="x", padx=12)

btn_row = tk.Frame(admin_card, bg=CARD)
btn_row.pack(fill="x", padx=14, pady=10)
GlowButton(btn_row, "Add Student", command=open_add_student_window,
           bg_color=CARD3, fg_color=ACCENT, width=130, height=38, icon="＋").pack(side="left", padx=(0,6))
GlowButton(btn_row, "Manage Students", command=open_admin_panel,
           bg_color=CARD3, fg_color=ACCENT2, width=152, height=38, icon="👥").pack(side="left", padx=(0,6))
GlowButton(btn_row, "Retrain Model", command=start_training,
           bg_color=CARD3, fg_color=ACCENT, width=138, height=38,
           icon="⚙", hover_color=CARD2).pack(side="right")

# ── Report ──
rep_card = tk.Frame(left_panel, bg=CARD,
                     highlightbackground=BORDER2, highlightthickness=1)
rep_card.pack(fill="x", pady=(8, 0))

SectionHeader(rep_card, "EXPORT REPORT", "📊", WARN).pack(
    anchor="w", padx=14, pady=(8, 4))
tk.Frame(rep_card, bg=BORDER, height=1).pack(fill="x", padx=12)

rep_row = tk.Frame(rep_card, bg=CARD)
rep_row.pack(fill="x", padx=14, pady=10)

date_wrap = tk.Frame(rep_row, bg=CARD2, highlightbackground=BORDER2, highlightthickness=1)
date_wrap.pack(side="left", padx=(0, 10))
tk.Label(date_wrap, text="DATE", fg=MUTED, bg=CARD2, font=FONT_TINY).pack(padx=10, pady=(4, 0))
cal = DateEntry(date_wrap, width=12, background=CARD3, foreground=TEXT,
                borderwidth=0, font=FONT_BODY)
cal.pack(padx=10, pady=(0, 6))

GlowButton(rep_row, "Export to Excel", command=download_report,
           bg_color=WARN, fg_color=BG, width=200, height=38, icon="↓").pack(side="left")

# ════════════════════════════════════════════════
#  MIDDLE PANEL  (recent activity)
# ════════════════════════════════════════════════
mid_panel = tk.Frame(root, bg=CARD, width=234,
                      highlightbackground=BORDER2, highlightthickness=1)
mid_panel.pack(side="left", fill="both", padx=(0, 6), pady=10)
mid_panel.pack_propagate(False)

mid_hdr = tk.Frame(mid_panel, bg=CARD2)
mid_hdr.pack(fill="x")
tk.Frame(mid_hdr, bg=ACCENT2, width=3).pack(side="left", fill="y")
tk.Label(mid_hdr, text="  RECENT ACTIVITY", fg=TEXT2, bg=CARD2,
         font=("Consolas", 8, "bold")).pack(side="left", pady=12)

tk.Label(mid_panel, text="Click entry to view details →",
         fg=MUTED, bg=CARD, font=("Segoe UI", 7, "italic")).pack(pady=(4, 2))
tk.Frame(mid_panel, bg=BORDER, height=1).pack(fill="x", padx=8)

canvas_mid = tk.Canvas(mid_panel, bg=CARD, highlightthickness=0)
scrollable_frame = tk.Frame(canvas_mid, bg=CARD)
scrollable_frame.bind("<Configure>",
    lambda e: canvas_mid.configure(scrollregion=canvas_mid.bbox("all")))
canvas_mid.create_window((0, 0), window=scrollable_frame, anchor="nw", width=226)
canvas_mid.pack(side="left", fill="both", expand=True, padx=4, pady=6)

# ════════════════════════════════════════════════
#  RIGHT PANEL  (student details)
# ════════════════════════════════════════════════
right_panel = tk.Frame(root, bg=CARD, width=320,
                        highlightbackground=BORDER2, highlightthickness=1)
right_panel.pack(side="right", fill="both", padx=(0, 10), pady=10)
right_panel.pack_propagate(False)

rp_hdr = tk.Frame(right_panel, bg=CARD2)
rp_hdr.pack(fill="x")
tk.Frame(rp_hdr, bg=ACCENT, width=3).pack(side="left", fill="y")
tk.Label(rp_hdr, text="  STUDENT DETAILS", fg=TEXT2, bg=CARD2,
         font=("Consolas", 8, "bold")).pack(side="left", pady=12)

tk.Frame(right_panel, bg=BORDER, height=1).pack(fill="x", padx=8)

detail_canvas = tk.Canvas(right_panel, bg=CARD, highlightthickness=0)
detail_content_frame = tk.Frame(detail_canvas, bg=CARD)
detail_content_frame.bind("<Configure>",
    lambda e: detail_canvas.configure(scrollregion=detail_canvas.bbox("all")))
detail_canvas.create_window((0, 0), window=detail_content_frame, anchor="nw", width=310)
detail_canvas.pack(side="left", fill="both", expand=True)

# Placeholder
ph_f = tk.Frame(detail_content_frame, bg=CARD)
ph_f.pack(expand=True, fill="both", pady=60)
tk.Label(ph_f, text="👆", fg=MUTED, bg=CARD, font=("Segoe UI", 28)).pack()
tk.Label(ph_f, text="Scan a student\nor click an activity",
         fg=MUTED, bg=CARD, font=("Segoe UI", 10), justify="center").pack(pady=6)

# ─────────────────────────────────────────────
#  CLOCK
# ─────────────────────────────────────────────
def update_clock():
    now = datetime.now()
    lbl_time.config(text=now.strftime("%H:%M:%S"))
    lbl_date.config(text=now.strftime("%a, %d %b %Y"))
    root.after(1000, update_clock)

update_clock()
update_dashboard()
root.mainloop()