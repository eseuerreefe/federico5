#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI de espectro de audio en tiempo real (bonita) para tu asistente por voz.
- Usa PyQtGraph + Qt para rendimiento y estética.
- Muestra: espectro en barras con pico retenido (peak hold), oscilograma, VU estéreo/mono,
  y estado de la máquina (PASSIVE / ACTIVE_LISTEN / PROCESSING) si lo integras.
- Se puede ejecutar standalone o integrarlo dentro de tu main loop.

Requisitos:
    pip install pyqt5 pyqtgraph sounddevice numpy

Ejecución directa:
    python spectrum_ui.py

Integración:
    1) Importa AudioSpectrumUI en tu script.
    2) Llama a ui = AudioSpectrumUI(sample_rate=16000, channels=1)
    3) ui.start() para abrir la ventana.
    4) ui.update_state("PASSIVE") cuando cambie tu FSM.
    5) ui.close() al salir.

Notas:
- Captura audio con sounddevice en modo callback para baja latencia.
- Dithering/averaging y smoothing exponencial para un movimiento agradable.
- Mapa logarítmico de frecuencia a 64 barras (ajustable).
- Tema oscuro con degradados y glow sutil.
"""
import sys, time, math, threading, collections
from typing import Deque, Optional

import numpy as np
import sounddevice as sd

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget
import pyqtgraph as pg


class AudioCapture:
    def __init__(self, samplerate: int = 16000, channels: int = 1, block_ms: int = 30):
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = int(samplerate * block_ms / 1000)
        self.q: Deque[np.ndarray] = collections.deque(maxlen=50)
        self.stream: Optional[sd.InputStream] = None
        self.lock = threading.Lock()
        self.running = False

    def _callback(self, indata, frames, time_info, status):
        if status:
            # status puede reportar under/overflows: lo ignoramos con gracia
            pass
        # Asegura 1 canal (si canales>1, mezcla a mono)
        data = indata.copy()
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        with self.lock:
            self.q.append(data.reshape(-1))

    def start(self, device=None):
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.samplerate,
            dtype='float32',
            blocksize=self.blocksize,
            callback=self._callback,
            device=device,
        )
        self.stream.start()

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.q:
                return self.q.popleft()
        return None

    def stop(self):
        self.running = False
        try:
            if self.stream:
                self.stream.stop(); self.stream.close()
        finally:
            self.stream = None


class AudioSpectrumUI(QWidget):
    def __init__(self, sample_rate=16000, channels=1, device=None, n_bars=64):
        super().__init__()
        self.setWindowTitle("Espectro de audio — Asistente")
        self.resize(1100, 650)

        # Captura
        self.cap = AudioCapture(sample_rate, channels)
        self.cap.start(device=device)

        # Parámetros DSP
        self.fs = sample_rate
        self.frame_len = 2048  # FFT base (potencia de 2)
        self.hop = 512
        self.window = np.hanning(self.frame_len).astype(np.float32)
        self.fft_bins = self.frame_len // 2
        self.min_db = -80.0
        self.max_db = 0.0
        self.n_bars = n_bars
        self.smoothing = 0.6  # 0..1 (más alto => más suave)

        # Buffers
        self.overlap = np.zeros(self.frame_len, dtype=np.float32)
        self.olap_filled = 0
        self.spec_smoothed = np.zeros(self.n_bars, dtype=np.float32)
        self.peak_vals = np.zeros(self.n_bars, dtype=np.float32)
        self.peak_fall = 0.015  # caída por tick

        # UI principal
        pg.setConfigOptions(antialias=True, background=(12, 12, 14), foreground=(230, 230, 235))

        layout = QVBoxLayout(self)
        self.top_bar = self._build_top_bar()
        layout.addLayout(self.top_bar)

        self.plots = QHBoxLayout()
        layout.addLayout(self.plots, 1)

        # Espectro (barras)
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setMenuEnabled(False)
        self.spec_plot.setMouseEnabled(x=False, y=False)
        self.spec_plot.hideAxis('bottom'); self.spec_plot.hideAxis('left')
        self.spec_plot.setYRange(self.min_db, self.max_db)
        self.spec_bars = pg.BarGraphItem(x=np.arange(self.n_bars), height=np.zeros(self.n_bars), width=0.8)
        self.spec_plot.addItem(self.spec_bars)
        # Pico retenido como puntos
        self.peak_scatter = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(250, 250, 250, 200))
        self.spec_plot.addItem(self.peak_scatter)
        self.plots.addWidget(self.spec_plot, 2)

        # Oscilograma
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setMenuEnabled(False)
        self.wave_plot.setMouseEnabled(x=False, y=False)
        self.wave_plot.hideAxis('bottom'); self.wave_plot.hideAxis('left')
        self.wave_curve = self.wave_plot.plot(np.zeros(1024))
        self.plots.addWidget(self.wave_plot, 1)

        # VU
        self.vu_plot = pg.PlotWidget()
        self.vu_plot.setMenuEnabled(False)
        self.vu_plot.setMouseEnabled(x=False, y=False)
        self.vu_plot.hideAxis('bottom'); self.vu_plot.hideAxis('left')
        self.vu_plot.setYRange(0, 1.0)
        self.vu_bar = pg.BarGraphItem(x=[0], height=[0.0], width=0.6)
        self.vu_plot.addItem(self.vu_bar)
        self.plots.addWidget(self.vu_plot, 0)

        # Timer de actualización
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000 // 60)  # ~60 FPS

        # Precalcula mapeo log-freq -> barras
        self._make_log_mapping()

        # Estado visible
        self.state = "PASSIVE"
        self.state_label.setText(self._state_text(self.state))

    def _build_top_bar(self):
        hb = QHBoxLayout()
        title = QLabel("<b>Espectro de audio</b>")
        title.setStyleSheet("font-size:18px; color:#e6e6eb;")
        hb.addWidget(title)
        hb.addStretch(1)
        self.state_label = QLabel("")
        self.state_label.setStyleSheet("padding:6px 10px; border-radius:10px; background:#1d1f24; color:#cbd5e1;")
        hb.addWidget(self.state_label)
        return hb

    def _state_text(self, st: str) -> str:
        color = {"PASSIVE":"#475569","ACTIVE_LISTEN":"#22c55e","PROCESSING":"#f59e0b"}.get(st, "#94a3b8")
        return f"<span style='color:{color}'>●</span> <b>{st}</b>"

    def update_state(self, st: str):
        self.state = st
        self.state_label.setText(self._state_text(st))

    def _make_log_mapping(self):
        # Bordes de banda logarítmicos entre 50 Hz y fs/2
        fmin = 50.0
        fmax = self.fs / 2.0
        edges = np.geomspace(fmin, fmax, self.n_bars + 1)
        freqs = np.fft.rfftfreq(self.frame_len, 1.0 / self.fs)
        self.band_idx = []
        for i in range(self.n_bars):
            lo, hi = edges[i], edges[i + 1]
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            if len(idx) == 0:
                idx = np.array([max(0, int((lo / fmax) * len(freqs)) )])
            self.band_idx.append(idx)

    def _tick(self):
        data = self.cap.read()
        if data is None:
            return
        # Construir ventana con solapamiento
        x = data.astype(np.float32)
        n = len(x)
        out = []
        i = 0
        while i < n:
            need = self.frame_len - self.olap_filled
            take = min(need, n - i)
            self.overlap[self.olap_filled:self.olap_filled + take] = x[i:i + take]
            self.olap_filled += take
            i += take
            if self.olap_filled >= self.frame_len:
                out.append(self.overlap.copy())
                # desplaza hop
                self.overlap[:-self.hop] = self.overlap[self.hop:]
                self.olap_filled = self.frame_len - self.hop
        if not out:
            # actualizar oscilograma/VU con lo que haya
            self._update_wave_vu(x)
            return

        # Procesa última ventana para espectro
        frame = out[-1] * self.window
        spec = np.fft.rfft(frame)
        mag = np.abs(spec) + 1e-12
        db = 20.0 * np.log10(mag)
        # Normaliza rango
        db = np.clip(db, self.min_db, self.max_db)

        # Reduce a n_bars por bandas log
        bars = np.zeros(self.n_bars, dtype=np.float32)
        for i, idx in enumerate(self.band_idx):
            if len(idx) == 0:
                bars[i] = self.min_db
            else:
                # media en dB (promedio de energía)
                lin = np.mean(10 ** (db[idx] / 20.0))
                bars[i] = 20 * np.log10(max(lin, 1e-6))

        # Suavizado exponencial
        self.spec_smoothed = (
            self.smoothing * self.spec_smoothed + (1 - self.smoothing) * bars
        )
        # Actualiza picos con caída
        self.peak_vals = np.maximum(self.peak_vals - self.peak_fall * (self.max_db - self.min_db), self.spec_smoothed)

        # Dibuja
        xs = np.arange(self.n_bars)
        # color único según energía media (0..1)
        norm = (self.spec_smoothed - self.min_db) / max(1e-6, (self.max_db - self.min_db))
        t = float(np.clip(norm.mean(), 0.0, 1.0))
        r = int(255 * max(0.0, min(1.0, 2*t - 0.2)))
        g = int(255 * max(0.0, min(1.0, 2*t)))
        b = int(255 * max(0.0, 1.2 - 2*t))

        self.spec_bars.setOpts(x=xs, height=self.spec_smoothed, width=0.9)
        self.spec_bars.setOpts(brush=pg.mkBrush(r, g, b, 220))

        self.peak_scatter.setData(xs, self.peak_vals)

        # Oscilograma/VU de la última porción
        self._update_wave_vu(x)


    def _update_wave_vu(self, x: np.ndarray):
        # Oscilograma (últimos 1024)
        if x.size < 1024:
            w = np.pad(x, (0, 1024 - x.size))
        else:
            w = x[-1024:]
        self.wave_curve.setData(w)
        # VU: RMS
        rms = float(np.sqrt(np.mean(w ** 2))) if w.size else 0.0
        vu = min(1.0, rms * 12)  # escalado visual
        self.vu_bar.setOpts(x=[0], height=[vu])

    def start(self):
        self.show()

    def closeEvent(self, e):
        self.cap.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    ui = AudioSpectrumUI(sample_rate=16000, channels=1)
    ui.start()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
