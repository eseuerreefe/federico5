# -*- coding: utf-8 -*-
import os, sys, time, tempfile, subprocess, json, traceback, socket, threading, queue, random, platform, re
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pydub.playback import play
from faster_whisper import WhisperModel
import requests
import unicodedata
from datetime import datetime 
import subprocess, sys, os, platform
import xml.etree.ElementTree as ET
from html import unescape


# =========================
# CONFIG
# =========================
SCRIPT_DIR = os.path.dirname(__file__)

SPEC_PROC = None  # proceso del espectro

DETACHED_PROCESS = 0x00000008


# SFX (asegúrate de que existen y tienen audio)
SFX_LISTEN_START = os.path.join(SCRIPT_DIR, "sonido1.wav")   # suena al detectar wake word
SFX_PROCESS_START = os.path.join(SCRIPT_DIR, "sonido2.wav")  # suena tras entender la orden (antes de pensar)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3:8b")
OLLAMA_CLI = os.environ.get("OLLAMA_CLI", "ollama")  # ruta a ollama.exe si no está en PATH
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "900"))  # seg (15 min para primera descarga)

SYSTEM_PROMPT = ("Responde en español.")

# TTS - EDGE
EDGE_TTS_VOICE = os.environ.get("EDGE_TTS_VOICE", "es-ES-ElviraNeural")
EDGE_TTS_RATE = os.environ.get("EDGE_TTS_RATE", "+0%")
EDGE_TTS_PITCH = os.environ.get("EDGE_TTS_PITCH", "+0Hz")
EDGE_TTS_VOLUME = os.environ.get("EDGE_TTS_VOLUME", "+0%")

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30
MAX_UTTERANCE_S = 20
SILENCE_END_MS = 500

# Dispositivo
INPUT_DEVICE_INDEX = None
AUTOCHOOSE_FOCUSRITE = True

# Wake word y estados
WAKE_WORD = "federico"
WAKE_ALIASES = [
    "federico", "fedérico", "fede rico", "fede", "fedeico", "federicoo",
    "fede-rico", "fede_rico", "Pederico", "de rico","Federico"
]
WAKE_TIMEOUT_S = 12.0                 # tiempo para decir la orden tras el beep
WAKE_COOLDOWN_S = 3.0                 # ventana tras beeps/TTS donde ignoramos wake word (anti-eco)
POST_LISTEN_BEEP_GUARD_MS = 350       # espera tras sonido1 antes de escuchar la orden (evitar que el beep entre)
STATE_PASSIVE = "PASSIVE"
STATE_ACTIVE_LISTEN = "ACTIVE_LISTEN"
STATE_PROCESSING = "PROCESSING"

# Localización por defecto para la meteo (puedes sobreescribir con variables de entorno)
DEFAULT_CITY = os.environ.get("CITY_NAME", "Bilbao")
DEFAULT_LAT  = float(os.environ.get("LAT", 43.2630))
DEFAULT_LON  = float(os.environ.get("LON", -2.9350))

# Depuración
DEBUG = True
def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs, flush=True)

# =========================
#alarma
# === ALARMA: cuenta atrás en nueva ventana CMD ===

def parse_duration_seconds(text: str, default_s: int = 300) -> int:
    # NO usamos _norm() porque elimina ":" y nos fastidia "1:30"
    def _sa(s: str) -> str:
        import unicodedata, re
        s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
        s = s.lower()
        return s

    t = _sa(text)

    # 1) mm:ss o hh:mm(:ss)
    m = re.search(r"\b(\d+)\s*:\s*(\d+)(?:\s*:\s*(\d+))?\b", t)
    if m:
        h, mnt, sec = 0, int(m.group(1)), int(m.group(2))
        if m.group(3):
            h = int(m.group(1)); mnt = int(m.group(2)); sec = int(m.group(3))
            return max(1, h*3600 + mnt*60 + sec)
        # si es mm:ss
        if mnt <= 59 and sec <= 59:
            return max(1, mnt*60 + sec)

    # 2) palabras → número
    WORDS = {
        "cero":0, "un":1, "una":1, "uno":1, "dos":2, "tres":3, "cuatro":4, "cinco":5,
        "seis":6, "siete":7, "ocho":8, "nueve":9, "diez":10, "once":11, "doce":12,
        "trece":13, "catorce":14, "quince":15, "veinte":20, "treinta":30, "cuarenta":40,
        "cincuenta":50, "sesenta":60
    }
    # manejos especiales
    t = re.sub(r"\bmedia\b", "0.5", t)   # media -> 0.5 (de hora/minuto)
    t = re.sub(r"\bmedio\b", "0.5", t)   # medio -> 0.5
    t = re.sub(r"\bcuarto\b", "0.25", t) # cuarto -> 0.25

    # reemplaza palabras-numero por dígitos
    def repl_wordnum(mo):
        w = mo.group(0)
        return str(WORDS.get(w, w))
    t = re.sub(r"\b(" + "|".join(sorted(WORDS.keys(), key=len, reverse=True)) + r")\b", repl_wordnum, t)

    # 3) captura segmentos (número) + unidad, sumando todos
    total_s = 0.0
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(horas?|hrs?|h|minutos?|mins?|m|segundos?|segs?|s)\b", t):
        val = float(num)
        if unit.startswith(("h","hr")):
            total_s += val * 3600
        elif unit.startswith(("m","min")):
            total_s += val * 60
        else:
            total_s += val

    # 3.1) “y medio / y media / y cuarto” después de “hora/minuto”
    # ej: "1 hora y media" -> ya tendremos 1h; añadimos 0.5h
    ym = re.search(r"(hora|minuto)s?\s+y\s+(0\.5|0\.25)\b", t)
    if ym:
        extra = float(ym.group(2))
        if "hora" in ym.group(1):
            total_s += extra * 3600
        else:
            total_s += extra * 60

    # 4) “en X” sin unidad -> minutos
    if total_s == 0:
        e = re.search(r"\ben\s+(\d+(?:\.\d+)?)\b", t)
        if e:
            total_s = float(e.group(1)) * 60

    # 5) número suelto -> minutos
    if total_s == 0:
        n = re.search(r"\b(\d+(?:\.\d+)?)\b", t)
        if n:
            total_s = float(n.group(1)) * 60

    if total_s <= 0:
        total_s = default_s

    return max(1, int(round(total_s)))


def is_alarma_intent(text: str) -> bool:
    """
    Detecta órdenes del tipo: 'ponme una alarma', 'temporizador', 'cuenta atrás', etc.
    """
    n = _norm(text)
    triggers = [
        "ponme una alarma", "pon una alarma", "programa una alarma", "alarma",
        "temporizador", "timer", "cuenta atras", "cuenta atrás",
        "despertador", "recordatorio en", "avísame en", "avisame en"
    ]
    return any(tr in n for tr in triggers)

# === ALARMA: cuenta atrás en un hilo (sin abrir CMD) ===
ALARM_WAV = os.path.join(SCRIPT_DIR, "alarma.wav")


def _alarm_worker(total_seconds: int):
    import time, os, platform
    total_seconds = max(1, int(total_seconds))
    print("=== Temporizador iniciado ===")
    t = total_seconds
    try:
        while t >= 0:
            m, s = divmod(t, 60)
            print(f"\r⏳ Cuenta atrás: {m:02d}:{s:02d}", end="", flush=True)
            time.sleep(1)
            t -= 1
        print("\n⏰ ¡Tiempo!")
    except KeyboardInterrupt:
        print("\n(Temporizador cancelado)")
        return

    try:
        if os.path.isfile(ALARM_WAV) and os.path.getsize(ALARM_WAV) > 0:
            play_sfx(ALARM_WAV, blocking=True)
        else:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 800)
            else:
                print("(No encontré alarma.wav)")
    except Exception as e:
        print("(No se pudo reproducir la alarma):", e)

def launch_alarm_countdown(total_seconds: int):
    # SOLO hilo en segundo plano, sin abrir nuevas ventanas
    threading.Thread(target=_alarm_worker, args=(total_seconds,), daemon=True).start()


# ===== TTS async + STOP =====
_CURRENT_TTS_HANDLE = None

def tts_stop():
    """Detiene cualquier audio TTS en curso (simpleaudio o winsound)."""
    global _CURRENT_TTS_HANDLE
    stopped = False
    try:
        # simpleaudio
        if _CURRENT_TTS_HANDLE is not None:
            try:
                _CURRENT_TTS_HANDLE.stop()
                stopped = True
            except Exception:
                pass
            _CURRENT_TTS_HANDLE = None
        # winsound (Windows): purga audio async
        if _HAS_WINSOUND:
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
                stopped = True
            except Exception:
                pass
    except Exception:
        pass
    return stopped

def tts_say_async(text):
    """Sintetiza a WAV y lo reproduce en background, guardando handle para tts_stop()."""
    out_path = f"reply_{int(time.time()*1000)}.wav"
    wav = (tts_with_edge_tts_wav(text, out_path) or
           tts_with_pyttsx3(text, out_path))
    if not wav or not os.path.exists(wav):
        print("[AudioOut] (Sin TTS) Respuesta:", text)
        return False

    # reproduce NO BLOQUEANTE y guarda handle
    global _CURRENT_TTS_HANDLE
    _CURRENT_TTS_HANDLE = None
    try:
        if _HAS_SIMPLEAUDIO:
            wave_obj = sa.WaveObject.from_wave_file(wav)
            _CURRENT_TTS_HANDLE = wave_obj.play()  # NO bloquea
        elif _HAS_WINSOUND:
            winsound.PlaySound(wav, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            # pydub en hilo aparte
            threading.Thread(target=speak_wav, args=(wav,), daemon=True).start()
        return True
    finally:
        # borra el WAV cuando termine (si usamos simpleaudio podemos esperar al terminar, pero lo dejamos)
        threading.Thread(target=lambda p: (time.sleep(10), os.path.exists(p) and os.remove(p)), args=(wav,), daemon=True).start()

#leer noticias
def is_noticias_intent(text: str) -> bool:
    n = _norm(text)
    return (
        "noticias" in n or
        "titulares" in n or
        "resumen de noticias" in n or
        "leer noticias" in n or
        "que hay de nuevo" in n or
        "qué hay de nuevo" in text.lower()
    )

def _rss_items(url, limit=5):
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        # Soporta RSS <channel><item> y Atom <feed><entry>
        chan = root.find("channel")
        if chan is not None:
            for it in chan.findall("item")[:limit]:
                tit = it.findtext("title") or ""
                tit = unescape(tit).strip()
                if tit:
                    items.append(tit)
        else:
            # Atom
            for it in root.findall("{http://www.w3.org/2005/Atom}entry")[:limit]:
                tit_el = it.find("{http://www.w3.org/2005/Atom}title")
                tit = (tit_el.text if tit_el is not None else "") or ""
                tit = unescape(tit).strip()
                if tit:
                    items.append(tit)
        return items
    except Exception:
        return []

def leer_noticias(limit_total=6) -> str:
    # Elige 2–3 feeds fiables (sin API key)
    feeds = [
        "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/portada",     # El País (RSS MRSS)
        "https://www.bbc.co.uk/mundo/ultimas_noticias/index.xml",               # BBC Mundo
        "https://e00-elmundo.uecdn.es/elmundo/rss/espana.xml",                  # El Mundo España
    ]
    titulares = []
    for f in feeds:
        if len(titulares) >= limit_total:
            break
        ts = _rss_items(f, limit=limit_total)  # pillamos varios, luego cortamos
        for t in ts:
            if t not in titulares:
                titulares.append(t)
            if len(titulares) >= limit_total:
                break

    if not titulares:
        return "No pude traer titulares ahora mismo."

    # Frase compacta para TTS
    frases = "; ".join(titulares[:limit_total])
    return f"Titulares: {frases}."



# “TÍO ENROLLADO”
# =========================
APERTURAS_COLEGAS = [
    "Buenos dias apuesto caballero  ",
]
def estilizar_respuesta(texto: str) -> str:
    apertura = random.choice(APERTURAS_COLEGAS)
    texto = (texto or "").strip()
    if not texto:
        return apertura.strip()
    return apertura + texto[0].upper() + texto[1:]


#espectro
def launch_spectrum():
    ui_path = os.path.join(SCRIPT_DIR, "spectrum_ui.py")
    if platform.system() == "Windows":
        return subprocess.Popen(
            [sys.executable, ui_path],
            creationflags=DETACHED_PROCESS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        return subprocess.Popen([sys.executable, ui_path])
# =========================
# WAV playback backends
# =========================
try:
    import simpleaudio as sa
    _HAS_SIMPLEAUDIO = True
except Exception:
    _HAS_SIMPLEAUDIO = False

try:
    import winsound  # nativo Windows
    _HAS_WINSOUND = (platform.system() == "Windows")
except Exception:
    _HAS_WINSOUND = False

def _play_wav_simpleaudio(path, blocking=False):
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        play_obj = wave_obj.play()
        if blocking:
            play_obj.wait_done()
        return True
    except Exception:
        return False

def _play_wav_winsound(path, blocking=False):
    try:
        flags = winsound.SND_FILENAME | (winsound.SND_SYNC if blocking else winsound.SND_ASYNC)
        winsound.PlaySound(path, flags)
        return True
    except Exception:
        return False

def _play_wav_pydub(path, blocking=False):
    try:
        seg = AudioSegment.from_wav(path)
        if blocking:
            play(seg)
        else:
            threading.Thread(target=play, args=(seg,), daemon=True).start()
        return True
    except Exception:
        return False

def play_sfx(path: str, blocking: bool = False):
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        dprint(f"[SFX] No se puede reproducir: {path}")
        return
    if _HAS_SIMPLEAUDIO and _play_wav_simpleaudio(path, blocking):
        return
    if _HAS_WINSOUND and _play_wav_winsound(path, blocking):
        return
    if _play_wav_pydub(path, blocking):
        return
    dprint(f"[SFX] Falló la reproducción: {path}")

def speak_wav(path):
    dprint(f"[AudioOut] Reproduciendo {path}…")
    ok = False
    if _HAS_SIMPLEAUDIO:
        ok = _play_wav_simpleaudio(path, blocking=True)
    if not ok and _HAS_WINSOUND:
        ok = _play_wav_winsound(path, blocking=True)
    if not ok:
        ok = _play_wav_pydub(path, blocking=True)
    dprint("[AudioOut] Reproducción terminada." if ok else "[AudioOut] Falló la reproducción.")

# =========================
# Checks / Ollama
# =========================
def is_port_open(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def check_ollama():
    try:
        from urllib.parse import urlparse
        u = urlparse(OLLAMA_URL)
        host = u.hostname or "127.0.0.1"
        port = u.port or 11434
        if not is_port_open(host, port, 0.8):
            print(f"[Check] Ollama puerto {host}:{port} -> NO ABIERTO")
            return {"api": False, "chat": False, "generate": False}
        r = requests.get(OLLAMA_URL.rstrip("/") + "/api/tags", timeout=3)
        print(f"[Check] Ollama /api/tags -> HTTP {r.status_code}")
        api_ok = r.ok
        ok_chat = ok_gen = False
        try:
            rq = requests.post(
                OLLAMA_URL.rstrip("/") + "/api/chat",
                json={"model": "tiny", "messages": [{"role": "user", "content": "ping"}], "stream": False},
                timeout=3,
            )
            ok_chat = rq.status_code != 404
        except Exception:
            ok_chat = False
        try:
            rg = requests.post(
                OLLAMA_URL.rstrip("/") + "/api/generate",
                json={"model": "tiny", "prompt": "ping", "stream": False},
                timeout=3,
            )
            ok_gen = rg.status_code != 404
        except Exception:
            ok_gen = False
        return {"api": api_ok, "chat": ok_chat, "generate": ok_gen}
    except Exception as e:
        print("[Check] Ollama no accesible:", repr(e))
        return {"api": False, "chat": False, "generate": False}

# =========================
# Dispositivos de audio
# =========================
def list_input_devices():
    dprint("[Audio] Enumerando dispositivos de entrada…")
    devices = sd.query_devices()
    idxs = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            idxs.append((i, d["name"]))
            dprint(f"  - idx {i}: {d['name']} | in_ch={d.get('max_input_channels')} | out_ch={d.get('max_output_channels')}")
    return idxs

def autodetect_focusrite():
    for i, name in list_input_devices():
        if "Focusrite" in name or "USB Audio" in name or "USB" in name:
            dprint(f"[Audio] Detectado candidato Focusrite/USB en idx {i} -> {name}")
            return i
    return None

def ensure_device():
    global INPUT_DEVICE_INDEX
    if INPUT_DEVICE_INDEX is None and AUTOCHOOSE_FOCUSRITE:
        idx = autodetect_focusrite()
        if idx is not None:
            INPUT_DEVICE_INDEX = idx
            print(f"[Audio] Usando dispositivo Focusrite detectado -> index {idx}")
        else:
            print("[Audio] No se detectó Focusrite automáticamente. Se usará el predeterminado del sistema.")
    else:
        print(f"[Audio] Usando dispositivo index={INPUT_DEVICE_INDEX} (si None, el predeterminado)")

def ensure_device_prints():
    try:
        default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        dprint(f"[Audio] Dispositivo por defecto de entrada en SD: {default_in}")
        _ = list_input_devices()
    except Exception:
        traceback.print_exc()

# =========================
# VAD y utilidades audio
# =========================
def rms(x):
    if len(x) == 0:
        return 0.0
    x = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(x))))

def calibrate_noise(stream, seconds=1.0):
    print("[VAD] Calibrando ruido ambiente…")
    frames_needed = int(seconds * SAMPLE_RATE)
    buf, samples_acc = [], 0
    while samples_acc < frames_needed:
        data, _ = stream.read(int(SAMPLE_RATE * 0.05))  # 50 ms
        buf.append(data.copy())
        samples_acc += data.shape[0] if hasattr(data, "shape") else len(data)
        if len(buf) % 2 == 0:
            audio_partial = np.concatenate(buf, axis=0).flatten()
            dprint(f"[VAD] Calibración progreso: {samples_acc}/{frames_needed} muestras, RMS parcial={rms(audio_partial):.4f}")
    audio = np.concatenate(buf, axis=0).flatten()
    baseline = rms(audio)
    threshold = max(0.008, baseline * 2.0)
    print(f"[VAD] Ruido: {baseline:.4f} | Umbral: {threshold:.4f}")
    return threshold

def collect_utterance(stream, threshold):
    """Devuelve np.array int16 con la locución detectada o None si no hay voz en ~5s."""
    frame_len = int(SAMPLE_RATE * (FRAME_MS/1000.0))
    silence_limit_frames = int((SILENCE_END_MS/1000.0) / (FRAME_MS/1000.0))
    max_frames = int((MAX_UTTERANCE_S) / (FRAME_MS/1000.0))

    th_on = threshold
    th_off = threshold * 0.6

    dprint(f"[VAD] Esperando voz: frame_len={frame_len}, silence_limit_frames={silence_limit_frames}, "
           f"max_frames={max_frames}, threshold_on={th_on:.4f}, threshold_off={th_off:.4f}")

    voiced = False
    voiced_frames = []
    silence_count = 0
    frame_count = 0
    start_time = time.time()

    while True:
        data, _ = stream.read(frame_len)
        mono = data[:, 0] if (hasattr(data, "ndim") and data.ndim > 1) else data
        level = rms(mono)
        frame_count += 1

        if frame_count % 5 == 0:
            bars = int(min(30, level * 2000))
            dprint(f"[VUM] |{'#'*bars}{'.'*(30-bars)}| lvl={level:.4f} thr_on={th_on:.4f} thr_off={th_off:.4f}")

        tag = "VOZ" if (level > th_on or (voiced and level >= th_off)) else "silencio"
        dprint(f"[VAD] frame={frame_count:05d} | nivel={level:.4f} | {tag} | silence_count={silence_count} | voiced={voiced}")

        if not voiced:
            if level > th_on:
                dprint("[VAD] >>> DETECTADO INICIO DE VOZ <<<")
                voiced = True
                voiced_frames.append(mono.copy())
                silence_count = 0
        else:
            if level >= th_off:
                silence_count = max(0, silence_count - 1)
            else:
                silence_count += 1
            voiced_frames.append(mono.copy())
            if silence_count >= silence_limit_frames:
                dprint("[VAD] Silencio suficientemente largo. Fin de locución.")
                break

        if len(voiced_frames) >= max_frames:
            dprint("[VAD] Cortado por duración máxima de locución.")
            break

        if not voiced and (time.time() - start_time) > 5.0:
            dprint("[VAD] 5s sin voz detectable. Devolviendo None para recalibrar.")
            return None

    if not voiced_frames:
        return None

    audio = np.concatenate(voiced_frames, axis=0).astype(np.int16)
    dprint(f"[VAD] Locución capturada: {len(audio)} muestras ({len(audio)/SAMPLE_RATE:.2f} s)")
    return audio

def write_wav_int16(path, audio_np):
    dprint(f"[I/O] Exportando WAV a {path}…")
    seg = AudioSegment(audio_np.tobytes(), frame_rate=SAMPLE_RATE, sample_width=2, channels=1)
    seg.export(path, format="wav")
    dprint("[I/O] WAV exportado.")

# =========================
# Whisper
# =========================
def transcribe_whisper(model, audio_np):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    write_wav_int16(wav_path, audio_np)
    dprint("[Whisper] Iniciando transcripción…")
    t0 = time.time()
    segments, info = model.transcribe(wav_path, language="es", beam_size=1, vad_filter=True)
    dprint(f"[Whisper] Transcripción completa en {time.time()-t0:.2f}s. {info}")
    text = "".join(s.text for s in segments).strip()
    dprint(f"[Whisper] Texto detectado (len={len(text)}): «{text}»")
    try:
        os.remove(wav_path)
        dprint("[I/O] WAV temporal eliminado.")
    except Exception:
        pass
    return text

# =========================
# Detección wake word (tolerante)
# =========================
def detect_wake_and_rest(text):
    """
    Devuelve (wake_detected: bool, resto: str) de forma tolerante:
    - ignora acentos, mayúsculas, signos y separadores
    - acepta alias (WAKE_ALIASES)
    - permite hasta 1 error de edición (levenshtein <= 1)
    - el 'resto' se corta del texto ORIGINAL, no del normalizado
    """
    if not text:
        return False, ""

    # --- helpers inlined ---
    def _strip_diacritics(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    def _normalize_for_match(s: str) -> str:
        s = _strip_diacritics(s.lower())
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    def _levenshtein_leq1(a: str, b: str) -> bool:
        if a == b:
            return True
        la, lb = len(a), len(b)
        if abs(la - lb) > 1:
            return False
        if la == lb:
            return sum(c1 != c2 for c1, c2 in zip(a, b)) <= 1
        if lb > la:
            a, b = b, a
            la, lb = lb, la
        i = j = diff = 0
        while i < la and j < lb:
            if a[i] == b[j]:
                i += 1; j += 1
            else:
                diff += 1; i += 1
                if diff > 1:
                    return False
        return True
    def _build_index_map(original: str):
        norm_chars, idx_map = [], []
        for i, c in enumerate(original):
            for c2 in unicodedata.normalize("NFKD", c).lower():
                if unicodedata.combining(c2):
                    continue
                if re.match(r"[a-z0-9]", c2):
                    norm_chars.append(c2); idx_map.append(i)
                else:
                    if not norm_chars or norm_chars[-1] != " ":
                        norm_chars.append(" "); idx_map.append(i)
        norm, mp = [], []
        for k, ch in enumerate(norm_chars):
            if ch == " " and (not norm or norm[-1] == " "):
                continue
            norm.append(ch); mp.append(idx_map[k])
        if norm and norm[0] == " ":
            norm = norm[1:]; mp = mp[1:]
        if norm and norm[-1] == " ":
            norm = norm[:-1]; mp = mp[:-1]
        return "".join(norm), mp
    # -----------------------

    norm, idx_map = _build_index_map(text)
    if not norm:
        return False, ""

    tokens = norm.split(" ")
    offs, pos = [], 0
    for t in tokens:
        start = norm.find(t, pos); end = start + len(t)
        offs.append((start, end)); pos = end

    norm_aliases = [_normalize_for_match(a) for a in WAKE_ALIASES]

    for idx_tok, tok in enumerate(tokens):
        for alias in norm_aliases:
            parts = alias.split(" ")
            L = len(parts)

            if L == 1:
                cand = tok
                if cand and _levenshtein_leq1(cand, parts[0]):
                    end_norm = offs[idx_tok][1]
                    end_orig = idx_map[end_norm - 1] + 1
                    rest = text[end_orig:].lstrip(" ,;:.-—–")
                    return True, rest
            else:
                if idx_tok + L <= len(tokens):
                    window = tokens[idx_tok:idx_tok + L]
                    diffs = 0; ok = True
                    for w, a in zip(window, parts):
                        if w == a:
                            continue
                        if not _levenshtein_leq1(w, a):
                            ok = False; break
                        diffs += 1
                        if diffs > 1:
                            ok = False; break
                    if ok:
                        end_norm = offs[idx_tok + L - 1][1]
                        end_orig = idx_map[end_norm - 1] + 1
                        rest = text[end_orig:].lstrip(" ,;:.-—–")
                        return True, rest
    return False, ""

#cerrar spoty
def is_cierra_spotify_intent(text: str) -> bool:
    n = _norm(text)
    return "cierra spotify" in n or "cerrar spotify" in n

def cerrar_spotify():
    try:
        if platform.system() == "Windows":
            os.system("taskkill /f /im Spotify.exe")
        elif platform.system() == "Linux":
            os.system("pkill spotify")
        elif platform.system() == "Darwin":  # macOS
            os.system("osascript -e 'quit app \"Spotify\"'")
        return "He cerrado Spotify, bro. Ahora silencio total."
    except Exception as e:
        return f"No pude cerrar Spotify: {e}"


# =========================
# Intención "haz la cama" + hora y meteo
# =========================
def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _norm(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

_HLC_PATTERNS = [
    "haz la cama", "has la cama", "hacer la cama", "hazme la cama", "haz la cama porfa",
    "as la cama", "hacerme la cama"
]

def _lev1(a: str, b: str) -> bool:
    if a == b: return True
    if abs(len(a)-len(b)) > 1: return False
    if len(a) == len(b):
        return sum(c1 != c2 for c1, c2 in zip(a, b)) <= 1
    if len(b) > len(a): a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            diff += 1; i += 1
            if diff > 1: return False
    return True

def is_haz_la_cama_intent(text: str) -> bool:
    n = _norm(text)
    for p in _HLC_PATTERNS:
        if _norm(p) in n:
            return True
    for p in _HLC_PATTERNS:
        if _lev1(_norm(p), n):
            return True
    return False
def is_listar_amigos_intent(text: str) -> bool:
    n = _norm(text)
    return "listar amigos" in n or "lista de amigos" in n

def is_callate_intent(text: str) -> bool:
    n = _norm(text)
    return (
        "callate" in n or "cállate" in text.lower() or
        "para" in n or "silencio" in n or "corta" in n or "stop" in n
    )


def listar_amigos() -> str:
    path = os.path.join(SCRIPT_DIR, "amigos.txt")
    if not os.path.exists(path):
        return "No encontré la lista de amigos, bro."
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return "La lista de amigos está vacía, parece que no tienes colegas jajaja."
        # Concatenamos para leerlo como speech
        salida = "Aquí tienes a tus colegas: " + ". ".join(lines)
        return salida
    except Exception as e:
        return f"No pude leer la lista de amigos: {e}"

def get_local_time_str() -> str:
    now = datetime.now()
    return now.strftime("%H:%M")

def fetch_weather_brief(lat=DEFAULT_LAT, lon=DEFAULT_LON, city=DEFAULT_CITY) -> str:
    """
    Usa Open-Meteo (sin API key). Devuelve breve: "En <ciudad>: XX°C, <cond>, máx/mín".
    Añade comentario extra en función de la temperatura actual.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,apparent_temperature,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
        }
        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()
        cur = data.get("current", {})
        daily = data.get("daily", {})
        t = cur.get("temperature_2m")
        sens = cur.get("apparent_temperature")
        wcode = int(cur.get("weather_code") or 0)
        tmax = (daily.get("temperature_2m_max") or [None])[0]
        tmin = (daily.get("temperature_2m_min") or [None])[0]

        WMAP = {
            0:"despejado", 1:"principalmente despejado", 2:"parcialmente nublado", 3:"nublado",
            45:"niebla", 48:"niebla escarchada", 51:"llovizna débil", 53:"llovizna",
            55:"llovizna fuerte", 61:"lluvia débil", 63:"lluvia", 65:"lluvia fuerte",
            71:"nieve débil", 73:"nieve", 75:"nieve fuerte", 80:"chubascos débiles",
            81:"chubascos", 82:"chubascos fuertes", 95:"tormenta", 96:"tormenta con granizo",
            99:"tormenta fuerte con granizo"
        }
        desc = WMAP.get(wcode, "tiempo estable")
        parts = []
        if t is not None: parts.append(f"{t:.0f}°C")
        parts.append(desc)
        if tmax is not None and tmin is not None:
            parts.append(f"siendo precise, tienes un máximo de {tmax:.0f} y un mínimo de {tmin:.0f}")
        if sens is not None:
            parts.append(f"hace una sensación termica de {sens:.0f} ,")

        base = f"En {city}: " + ", ".join(parts)

        # --- comentario gracioso según temperatura ---
        comentario = ""
        if t is not None:
            if t < 10:
                comentario = " yo que tú me abrigaba bro."
            elif t < 15:
                comentario = " Saca una sudadera no seas morau que tampoco hace tanto calor."
            elif t < 20:
                comentario = " Hace mítico día normal en el País Vasco, ni te rayes."
            elif t < 25:
                comentario = " Ojito que acecha el calor, sácate un pantalón corto para estar a gusto y atraer a las nenas con esas piernas sexys que me llevas guapetón, que te lo como todo."
            else:
                comentario = " Bro hace un calor de atar, vete a surfear que solo se está a gusto en el agua jajaja."

        return base + comentario
    except Exception:
        return "No pude consultar el tiempo ahora mismo."


# =========================
# Ollama (API + CLI fallback)
# =========================
def _messages_to_prompt(messages):
    parts = []
    system = next((m["content"] for m in messages if m["role"] == "system"), None)
    if system: parts.append(f"<<SYS>>\n{system}\n<</SYS>>")
    for m in messages:
        if m["role"] == "user":
            parts.append(f"Usuario: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"Asistente: {m['content']}")
    parts.append("Asistente:")
    return "\n".join(parts)

def chat_ollama_api(messages, prefer="chat"):
    def call_chat():
        url = OLLAMA_URL.rstrip("/") + "/api/chat"
        payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0.5}}
        dprint(f"[Ollama] POST {url} | modelo={OLLAMA_MODEL}")
        r = requests.post(url, json=payload, timeout=(5, 60))
        if r.status_code == 404:
            raise FileNotFoundError("chat 404")
        r.raise_for_status()
        data = r.json()
        return (data.get("message", {}) or {}).get("content", "").strip()

    def call_generate():
        url = OLLAMA_URL.rstrip("/") + "/api/generate"
        payload = {"model": OLLAMA_MODEL, "prompt": _messages_to_prompt(messages), "stream": False, "options": {"temperature": 0.5}}
        dprint(f"[Ollama] POST {url} | modelo={OLLAMA_MODEL}")
        r = requests.post(url, json=payload, timeout=(5, 60))
        if r.status_code == 404:
            raise FileNotFoundError("generate 404")
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    try:
        if prefer == "chat":
            return call_chat()
        else:
            return call_generate()
    except FileNotFoundError:
        try:
            return call_generate() if prefer == "chat" else call_chat()
        except Exception as e2:
            raise e2

def _iter_stream(pipe, out_queue, tag):
    for line in iter(pipe.readline, b''):
        try:
            out_queue.put((tag, line.decode("utf-8", errors="ignore")))
        except Exception:
            out_queue.put((tag, line))
    pipe.close()

def chat_ollama_cli(messages):
    prompt = _messages_to_prompt(messages)
    exe = OLLAMA_CLI
    cmd = [exe, "run", OLLAMA_MODEL]
    dprint(f"[Ollama CLI] Ejecutando: {' '.join(cmd)}")
    try:
        p = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False
        )
    except FileNotFoundError:
        dprint("[Ollama CLI] No encontrado. Ajusta OLLAMA_CLI con la ruta a ollama.exe")
        raise

    qlines = queue.Queue()
    t_out = threading.Thread(target=_iter_stream, args=(p.stdout, qlines, "OUT"), daemon=True)
    t_err = threading.Thread(target=_iter_stream, args=(p.stderr, qlines, "ERR"), daemon=True)
    t_out.start(); t_err.start()

    try:
        p.stdin.write(prompt.encode("utf-8"))
        p.stdin.close()
    except Exception:
        pass

    start = time.time()
    chunks_out = []
    shown_pull_hint = False

    while True:
        try:
            tag, line = qlines.get(timeout=0.2)
            if tag == "ERR":
                text = line.strip()
                if text:
                    if ("pulling" in text.lower() or "%" in text) and not shown_pull_hint:
                        print("[Ollama CLI] Descargando modelo (esto puede tardar la primera vez)…")
                        shown_pull_hint = True
                    dprint(f"[Ollama CLI][stderr] {text}")
            else:
                if line:
                    chunks_out.append(line)
        except queue.Empty:
            pass

        if p.poll() is not None:
            break

        if (time.time() - start) > OLLAMA_TIMEOUT:
            try:
                p.kill()
            except Exception:
                pass
            raise TimeoutError("Ollama CLI tardó demasiado (timeout). Prueba a predescargar el modelo con: ollama pull " + OLLAMA_MODEL)

    while True:
        try:
            tag, line = qlines.get_nowait()
            if tag == "OUT":
                chunks_out.append(line)
            else:
                dprint(f"[Ollama CLI][stderr] {line.strip()}")
        except queue.Empty:
            break

    out = "".join(chunks_out).strip()
    if not out:
        out = "No recibí respuesta del CLI de Ollama."
    return out

def get_reply_ollama(messages, caps):
    if caps.get("api"):
        try:
            prefer = "chat" if caps.get("chat") else "generate"
            return chat_ollama_api(messages, prefer=prefer)
        except Exception:
            dprint("[Ollama] API falló, probando CLI…")
    return chat_ollama_cli(messages)

def chat_fallback(messages):
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    return f"Hola. No puedo conectar con el modelo local ahora mismo. Dijiste: «{last_user}». ¿Intento de nuevo?"

# =========================
# TTS (EDGE -> WAV PCM | fallback pyttsx3)
# =========================
def tts_with_edge_tts_wav(text, out_path_wav, voice=EDGE_TTS_VOICE):
    try:
        cmd = [
            sys.executable, "-m", "edge_tts",
            "--voice", voice,
            "--text", text,
            "--format", "riff-16khz-16bit-mono-pcm",
            "--write-media", out_path_wav,
            "--rate", EDGE_TTS_RATE,
            "--pitch", EDGE_TTS_PITCH,
            "--volume", EDGE_TTS_VOLUME,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return out_path_wav if os.path.isfile(out_path_wav) and os.path.getsize(out_path_wav) > 0 else None
    except subprocess.CalledProcessError as e:
        print("[edge-tts] STDERR:", e.stderr.decode(errors="ignore"))
        return None
    except Exception:
        traceback.print_exc()
        return None

def tts_with_pyttsx3(text, out_path_wav):
    try:
        import pyttsx3
    except Exception:
        return None
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 180)
        engine.save_to_file(text, out_path_wav)
        engine.runAndWait()
        return out_path_wav if os.path.exists(out_path_wav) and os.path.getsize(out_path_wav) > 0 else None
    except Exception:
        traceback.print_exc()
        return None

def tts_say_blocking(text):
    out_path = f"reply_{int(time.time()*1000)}.wav"
    wav = (tts_with_edge_tts_wav(text, out_path) or
           tts_with_pyttsx3(text, out_path))
    if wav and os.path.exists(wav):
        try:
            speak_wav(wav)
        finally:
            try:
                os.remove(wav)
            except Exception:
                pass
    else:
        print("[AudioOut] (Sin TTS) Respuesta:", text)

def is_subeme_animo_intent(text: str) -> bool:
    n = _norm(text)
    return "subeme el animo" in n or "súbeme el ánimo" in n or "suve el ánimo" in n or "sube el ánimo" in n

def subeme_animo():
    uri = "spotify:track:4LsYdWDeumtYjMndQVcA94"
    try:
        os.startfile(uri)  # Windows abre directamente la app asociada a "spotify:"
        return "Bro, te pongo un temazo en Spotify para que se te suba el ánimo!"
    except Exception as e:
        return f"No pude lanzar Spotify: {e}"

#fraseslegendarias
# === FRASES LEGENDARIAS ===

def is_apunta_frase_intent(text: str) -> bool:
    n = _norm(text)
    return n.startswith("apunta esta frase") or n.startswith("guarda esta frase")

def is_dime_frase_intent(text: str) -> bool:
    n = _norm(text)
    return "dime una frase legendaria" in n or "frase legendaria" in n

def guardar_frase(text: str) -> str:
    path = os.path.join(SCRIPT_DIR, "frases.txt")
    n = _norm(text)

    # eliminamos el trigger "apunta esta frase" o "guarda esta frase"
    frase = re.sub(r"^(apunta|guarda)\s+esta\s+frase( legendaria)?", "", n, flags=re.IGNORECASE).strip()

    if not frase:
        return "No pillé bien la frase, repítela crack."
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {frase}\n")
        return f"He guardado tu frase legendaria: «{frase}»"
    except Exception as e:
        return f"No pude guardar la frase: {e}"


def leer_frase() -> str:
    path = os.path.join(SCRIPT_DIR, "frases.txt")
    if not os.path.exists(path):
        return "Todavía no tienes frases legendarias, empieza a soltar sabiduría bro."
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return "El archivo está vacío, empieza a dejar tu huella tío."
        return random.choice(lines)
    except Exception as e:
        return f"No pude leer tus frases: {e}"



# =========================
# MAIN LOOP — Máquina de estados
# =========================
def main():
    print("=== Chequeo de dependencias ===")
    ollama_caps = check_ollama()
    print("===============================\n")

    ensure_device()
    ensure_device_prints()


    # >>> abre la UI del espectro <<<
    global SPEC_PROC
    SPEC_PROC = launch_spectrum()
    # (opcional) dale un pelín de tiempo a que abra
    time.sleep(0.3)

    print("[Whisper] Cargando modelo…")
    whisper = WhisperModel("small", device="cpu", compute_type="int8")
    dprint("[Whisper] Modelo cargado OK.")

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"\n🤖 Modo pasivo. Di «{WAKE_WORD}» para activarme. Ctrl+C para salir.\n")

    blocksize = int(SAMPLE_RATE * (FRAME_MS/1000.0))
    dprint(f"[Audio] Abriendo InputStream(sample_rate={SAMPLE_RATE}, channels={CHANNELS}, dtype=int16, "
           f"blocksize={blocksize}, device={INPUT_DEVICE_INDEX})")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16",
                        blocksize=blocksize, device=INPUT_DEVICE_INDEX) as stream:
        dprint("[Audio] InputStream abierto.")
        threshold = calibrate_noise(stream)

        state = STATE_PASSIVE
        suppress_wake_until = 0.0          # epoch: durante esta ventana ignoramos wake word
        active_deadline = 0.0              # límite para decir la orden en ACTIVE_LISTEN

        while True:
            try:
                now = time.time()

                if state == STATE_PASSIVE:
                    # Solo detectamos wake word si no estamos en cooldown
                    utter = collect_utterance(stream, threshold)
                    if utter is None:
                        print("[Wake] No hubo voz. Recalibrando umbral…")
                        threshold = calibrate_noise(stream)
                        continue

                    # Transcripción
                    try:
                        write_wav_int16("wake_candidate.wav", utter)
                    except Exception:
                        pass
                    text = transcribe_whisper(whisper, utter)
                    dprint(f"[Wake] Candidato: «{text}»")

                    if now < suppress_wake_until:
                        dprint(f"[Wake] En cooldown, ignorando posibles '{WAKE_WORD}'.")
                        continue

                    is_wake, rest = detect_wake_and_rest(text)

                    if is_wake and is_callate_intent(rest):
                        if tts_stop():
                            print("[TTS] Parado por orden 'cállate'.")
                        else:
                            print("[TTS] No había TTS en curso.")
                        # seguimos en PASSIVE
                        suppress_wake_until = time.time() + 0.5
                        continue

                    if is_callate_intent(text):
                        if tts_stop():
                            print("[TTS] Parado por orden 'cállate'.")
                        suppress_wake_until = time.time() + 0.5
                        continue
                        
                    if not is_wake:
                        dprint(f"[Wake] No se detectó '{WAKE_WORD}'. Sigo pasivo.")
                        continue

                    # Wake detectado
                    dprint(f"[Wake] >>> DETECTADO '{WAKE_WORD}' <<<")
                    play_sfx(SFX_LISTEN_START, blocking=False)
                    suppress_wake_until = time.time() + WAKE_COOLDOWN_S

                    # Si hay orden en la misma frase, procesarla ya
                    if rest and rest.strip():
                        user_text = rest.strip()
                        print(f"[Wake] Orden inline tras wake word: «{user_text}»")
                        state = STATE_PROCESSING
                    else:
                        # Cambiamos a estado de escucha activa
                        print("[Wake] Te escucho la orden…")
                        time.sleep(POST_LISTEN_BEEP_GUARD_MS / 1000.0)  # evitar que el beep contamine la captura
                        active_deadline = time.time() + WAKE_TIMEOUT_S
                        state = STATE_ACTIVE_LISTEN
                        continue  # siguiente iteración

                if state == STATE_ACTIVE_LISTEN:
                    # Aquí NO buscamos wake word, solo esperamos la orden
                    if time.time() > active_deadline:
                        print("[Wake] Timeout esperando la orden. Vuelvo a pasivo.")
                        state = STATE_PASSIVE
                        continue

                    utter = collect_utterance(stream, threshold)
                    if utter is None:
                        # 5s sin voz durante ACTIVE => recalibramos rápido y seguimos
                        threshold = calibrate_noise(stream, seconds=0.5)
                        continue

                    try:
                        write_wav_int16("command_utterance.wav", utter)
                    except Exception:
                        pass

                    user_text = transcribe_whisper(whisper, utter)
                    if not user_text.strip():
                        print("[Wake] No entendí claramente la orden. Sigue hablando…")
                        continue

                    dprint(f"[Wake] Orden capturada: «{user_text}»")
                    state = STATE_PROCESSING

                if state == STATE_PROCESSING:
                    # Aviso: empezamos a procesar (beep 2)
                    print("📝 Entendido. Procesando…")
                    play_sfx(SFX_PROCESS_START, blocking=False)

                    print(f"Tú: {user_text}")
                    history.append({"role": "user", "content": user_text})

                    # === ATAJO: "listar amigos" ===
                    if is_listar_amigos_intent(user_text):
                        reply = listar_amigos()
                        reply = estilizar_respuesta(reply)

                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})

                        print("🗣️ Hablando…")
                        tts_say_async(reply)


                        # Anti-eco: tras hablar, ignorar wake word un rato
                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S

                        # Podar historial
                        if len(history) > 16:
                            history = [history[0]] + history[-14:]
                            dprint("[Hist] Historial recortado.")

                        # Vuelve a pasivo
                        print(f"\n🤖 Vuelvo a modo pasivo. Di «{WAKE_WORD}» para activarme.\n")
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===
                    #NOTICIAS
                    # === ATAJO: "leer noticias" ===
                    if is_noticias_intent(user_text):
                        reply = leer_noticias(limit_total=6)
                        reply = estilizar_respuesta(reply)

                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})

                        print("🗣️ Hablando…")
                        tts_say_async(reply)

                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S

                        if len(history) > 16:
                            history = [history[0]] + history[-14:]
                            dprint("[Hist] Historial recortado.")

                        print(f"\n🤖 Vuelvo a modo pasivo. Di «{WAKE_WORD}» para activarme.\n")
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===
                    #alarma
                    # === ATAJO: "alarma / temporizador" ===
                    if is_alarma_intent(user_text):
                        secs = parse_duration_seconds(user_text, default_s=300)  # por defecto 5 min
                        mins = secs // 60
                        reply = f"Te pongo un temporizador de {mins} minutos y {secs%60} segundos. Abro una ventanita con la cuenta atrás y te aviso al terminar."
                        reply = estilizar_respuesta(reply)

                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})

                        print("🗣️ Hablando…")
                        tts_say_async(reply)

                        # Lanzar el temporizador en nueva ventana
                        launch_alarm_countdown(secs)

                        # Anti-eco y vuelta a pasivo
                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S
                        if len(history) > 16:
                            history = [history[0]] + history[-14:]
                            dprint("[Hist] Historial recortado.")
                        print(f"\n🤖 Vuelvo a modo pasivo. Di «{WAKE_WORD}» para activarme.\n")
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===




                    # === ATAJO: "subeme el animo" ===
                    if is_subeme_animo_intent(user_text):
                        reply = subeme_animo()
                        reply = estilizar_respuesta(reply)

                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})

                        print("🗣️ Hablando…")
                        tts_say_async(reply)


                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S

                        if len(history) > 16:
                            history = [history[0]] + history[-14:]
                            dprint("[Hist] Historial recortado.")

                        print(f"\n🤖 Vuelvo a modo pasivo. Di «{WAKE_WORD}» para activarme.\n")
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===



                    # === ATAJO: "haz la cama" -> hora + tiempo ===
                    if is_haz_la_cama_intent(user_text):
                        hora = get_local_time_str()
                        meteo = fetch_weather_brief()
                        reply = f"Son las {hora}. {meteo}"
                        reply = estilizar_respuesta(reply)

                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})

                        print("🗣️ Hablando…")
                        tts_say_async(reply)


                        # Anti-eco: tras hablar, ignorar wake word un rato
                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S

                        # Podar historial
                        if len(history) > 16:
                            history = [history[0]] + history[-14:]
                            dprint("[Hist] Historial recortado.")

                        # Volvemos al estado pasivo
                        print(f"\n🤖 Vuelvo a modo pasivo. Di «{WAKE_WORD}» para activarme.\n")
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===

                    #cerrar spoty
                    # === ATAJO: "cierra spotify" ===
                    if is_cierra_spotify_intent(user_text):
                        reply = cerrar_spotify()
                        reply = estilizar_respuesta(reply)

                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})

                        print("🗣️ Hablando…")
                        tts_say_async(reply)


                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S

                        if len(history) > 16:
                            history = [history[0]] + history[-14:]
                            dprint("[Hist] Historial recortado.")

                        print(f"\n🤖 Vuelvo a modo pasivo. Di «{WAKE_WORD}» para activarme.\n")
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===
                    # === ATAJO: "apunta esta frase" ===
                    if is_apunta_frase_intent(user_text):
                        reply = guardar_frase(user_text)
                        reply = estilizar_respuesta(reply)
                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})
                        print("🗣️ Hablando…")
                        tts_say_async(reply)

                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===

                    # === ATAJO: "dime una frase legendaria" ===
                    if is_dime_frase_intent(user_text):
                        reply = leer_frase()
                        reply = estilizar_respuesta(reply)
                        print(f"Asistente: {reply}")
                        history.append({"role": "assistant", "content": reply})
                        print("🗣️ Hablando…")
                        tts_say_async(reply)
                        suppress_wake_until = time.time() + WAKE_COOLDOWN_S
                        state = STATE_PASSIVE
                        continue
                    # === FIN ATAJO ===


                    print("🧠 Pensando con Ollama…")
                    try:
                        reply = get_reply_ollama(history, ollama_caps) if ollama_caps.get("api") else chat_ollama_cli(history)
                    except Exception:
                        print("[Ollama] Error llamando a Ollama, uso fallback simple.")
                        traceback.print_exc()
                        reply = chat_fallback(history)

                    if not reply:
                        reply = "No obtuve respuesta del modelo."

                    # Estilo “tío enrollado”
                    reply = estilizar_respuesta(reply)

                    print(f"Asistente: {reply}")
                    history.append({"role": "assistant", "content": reply})

                    print("🗣️ Hablando…")
                    tts_say_async(reply)


                    # Anti-eco: tras hablar, ignorar wake word un rato
                    suppress_wake_until = time.time() + WAKE_COOLDOWN_S

                    # Podar historial
                    if len(history) > 16:
                        history = [history[0]] + history[-14:]
                        dprint("[Hist] Historial recortado.")

                    # Volvemos al estado pasivo
                    print(f"\n🤖 Vuelvo a modo pasivo. Di «{WAKE_WORD}» para activarme.\n")
                    state = STATE_PASSIVE
                    continue

            except KeyboardInterrupt:
                print("\nHasta luego 👋")
                try:
                    if SPEC_PROC and SPEC_PROC.poll() is None:
                        SPEC_PROC.terminate()
                except Exception:
                    pass
                break
            except Exception:
                print("[Main] Error inesperado en el loop:")
                traceback.print_exc()
                time.sleep(0.5)

if __name__ == "__main__":
    main()
