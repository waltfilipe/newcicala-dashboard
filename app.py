import re
import os
import math
import tempfile
from pathlib import Path
from io import BytesIO
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
import mplcursors

# PDF
PDF_AVAILABLE = True
try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except Exception:
    PDF_AVAILABLE = False

# PAGE CONFIG
st.set_page_config(layout="wide", page_title="Cicala — Season Dashboard")

# OPTIONAL DOCX IMPORT
DOCX_AVAILABLE = True
try:
    from docx import Document
except Exception:
    DOCX_AVAILABLE = False

# STYLE
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(30,30,46,0.9), rgba(20,20,32,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 12px 16px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# CONSTANTS
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
LANE_LEFT_MIN = 53.33
LANE_RIGHT_MAX = 26.67
GOAL_X = 120.0
GOAL_Y = 40.0
FIG_W, FIG_H = 7.0, 4.7
FIG_DPI = 180
COLOR_SUCCESS = "#c8c8c8"
COLOR_PROGRESSIVE = "#2F80ED"
COLOR_FAIL = "#E07070"
ALPHA_SUCCESS = 0.07
C_BLUE = "#2F80ED"
C_BLUE_DARK = "#1a56db"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_PURPLE_LIGHT = "#a78bfa"
C_BLUE_PASTEL = "#5b9bd5"
C_GREEN_PASTEL = "#70ad47"
C_AMBER_PASTEL = "#d4a843"
PASS_TONES = ["#5b9bd5", "#3b82f6", "#1d4ed8"]
GRAY_TONES = ["#b8c0cc", "#8b93a7", "#6b7280"]
DEF_TONES = ["#5b9bd5", "#3b82f6", "#1d4ed8"]
CMAP_TOP10 = LinearSegmentedColormap.from_list("top10", ["#fef08a", "#f97316", "#b91c1c"])
NORM_TOP10 = Normalize(vmin=0.05, vmax=0.40)
NX_XT, NY_XT = 16, 12
D_REF, D_SCALE, BONUS_CAP = 10.0, 20.0, 0.60
LATERAL_MIN_DIST = 12.0
PENALTY_AREA_X = 18.0
FUNNEL_X_EXTEND = 33.0
PENALTY_AREA_Y_MIN = 18.0
PENALTY_AREA_Y_MAX = 62.0

def _hex_to_rgba(hex_color, alpha=1.0):
    if hex_color.startswith('#'):
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'rgba({r},{g},{b},{alpha})'
    return hex_color

def get_lane(y):
    if y >= LANE_LEFT_MIN:
        return "left"
    elif y < LANE_RIGHT_MAX:
        return "right"
    return "center"

def distance_to_goal(x, y):
    return np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2)

def is_progressive_pass(x_start, y_start, x_end, y_end):
    if x_start < 35:
        return False
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    if start_dist == 0:
        return False
    return ((start_dist - end_dist) / start_dist) >= 0.25

def classify_pass_direction(x_start, y_start, x_end, y_end):
    dx = x_end - x_start
    dy = y_end - y_start
    dist = np.sqrt(dx ** 2 + dy ** 2)
    angle_deg = np.degrees(np.arctan2(abs(dy), dx))
    if angle_deg <= 45.0:
        return "forward"
    if angle_deg >= 135.0:
        return "backward"
    if dist > LATERAL_MIN_DIST:
        return "lateral_right" if dy > 0 else "lateral_left"
    return "forward" if dx >= 0 else "backward"

def distance_bonus(distance):
    excess = np.maximum(0.0, np.asarray(distance, dtype=float) - D_REF)
    return np.minimum(BONUS_CAP, np.log1p(excess / D_SCALE))

@st.cache_data(show_spinner=False)
def compute_xt_grid(NX=16, NY=12, sub=24):
    ncols_hr = NX * sub
    nrows_hr = NY * sub
    xe = np.linspace(0, FIELD_X, ncols_hr + 1)
    ye = np.linspace(0, FIELD_Y, nrows_hr + 1)
    xc = (xe[:-1] + xe[1:]) / 2
    yc_arr = (ye[:-1] + ye[1:]) / 2
    Xc, Yc = np.meshgrid(xc, yc_arr)
    xp = 0.01 + (Xc / FIELD_X) * 0.99
    yc = 1.0 - np.abs((Yc / FIELD_Y) - 0.5) * 2.0
    base = xp * (0.8 + 0.2 * yc)
    base = (base - base.min()) / (base.max() - base.min() + 1e-12)
    XT = base.copy()
    XT = (XT - XT.min()) / (XT.max() - XT.min() + 1e-12)
    XTc = np.zeros((NY, NX))
    for iy in range(NY):
        for ix in range(NX):
            XTc[iy, ix] = XT[iy * sub:(iy + 1) * sub, ix * sub:(ix + 1) * sub].mean()
    XTc = (XTc - XTc.min()) / (XTc.max() - XTc.min() + 1e-12)
    return XTc

XT_GRID = compute_xt_grid()

def xt_value(x, y):
    ix = int(np.clip((x / FIELD_X) * NX_XT, 0, NX_XT - 1))
    iy = int(np.clip((y / FIELD_Y) * NY_XT, 0, NY_XT - 1))
    return float(XT_GRID[iy, ix])

def is_in_funnel_zone(x, y):
    return x <= FUNNEL_X_EXTEND and PENALTY_AREA_Y_MIN <= y <= PENALTY_AREA_Y_MAX

# ── DATA ───────────────────────────────────────────────────────
BASE_MATCHES_DATA = {
    "Connecticut United (03-27)": [
        ("PASS WON", 26.75, 68.34, 8.97, 51.05, None),
        ("PASS WON", 31.24, 51.22, 34.57, 72.50, None),
        ("PASS WON", 36.06, 46.90, 44.37, 57.04, None),
        ("PASS WON", 48.36, 64.02, 58.17, 51.72, None),
        ("PASS WON", 58.17, 64.02, 62.49, 55.21, None),
        ("PASS WON", 54.51, 49.72, 64.82, 61.69, None),
        ("PASS WON", 42.21, 70.84, 34.90, 76.49, None),
        ("PASS WON", 43.54, 75.32, 36.73, 67.84, None),
        ("PASS WON", 32.24, 53.96, 6.81, 38.50, None),
        ("PASS WON", 33.57, 65.77, 36.56, 75.57, None),
        ("PASS WON", 37.39, 61.11, 43.04, 75.41, None),
        ("PASS WON", 65.49, 53.63, 56.18, 70.42, None),
        ("PASS WON", 55.68, 48.15, 46.87, 30.86, None),
        ("PASS WON", 52.02, 22.05, 46.70, 41.99, None),
        ("PASS WON", 62.16, 35.51, 71.80, 35.18, None),
        ("PASS WON", 54.02, 33.35, 63.99, 22.55, None),
        ("PASS WON", 60.00, 22.21, 76.62, 32.85, None),
        ("PASS WON", 87.10, 9.41, 77.45, 16.23, None),
        ("PASS WON", 62.66, 20.05, 117.18, 8.25, None),
        ("PASS WON", 98.90, 43.49, 103.22, 47.15, None),
        ("PASS WON", 70.31, 45.98, 82.28, 60.11, None),
        ("PASS WON", 85.10, 75.24, 101.39, 74.08, None),
        ("PASS WON", 53.18, 67.59, 39.05, 59.62, None),
        ("PASS WON", 55.18, 49.64, 54.85, 13.07, None),
        ("PASS WON", 68.64, 19.22, 49.03, 24.37, None),
        ("PASS WON", 53.35, 22.71, 59.34, 30.19, None),
        ("PASS WON", 44.37, 24.71, 40.05, 46.82, None),
        ("PASS WON", 43.88, 39.34, 41.38, 73.08, None),
        ("PASS WON", 56.84, 53.46, 70.81, 76.24, None),
        ("PASS WON", 82.77, 12.24, 91.42, 4.59, None),
        ("PASS WON", 108.04, 11.74, 115.69, 58.29, None),
        ("PASS WON", 93.08, 3.93, 111.03, 13.74, None),
        ("PASS WON", 84.60, 17.89, 96.74, 22.05, None),
        ("PASS WON", 58.34, 16.06, 65.65, 2.43, None),
        ("PASS WON", 52.02, 8.58, 44.37, 15.73, None),
        ("PASS WON", 61.00, 23.21, 49.36, 15.23, None),
        ("PASS WON", 32.74, 30.69, 50.03, 33.02, None),
        ("PASS WON", 51.85, 33.68, 60.66, 40.00, None),
        ("PASS WON", 79.95, 60.45, 98.23, 60.28, None),
        ("PASS WON", 31.24, 52.14, 39.05, 72.08, None),
        ("PASS WON", 39.72, 48.98, 33.40, 57.62, None),
        ("PASS WON", 70.64, 51.47, 61.00, 51.64, None),
        ("PASS LOST", 53.35, 19.55, 73.96, 11.24, None),
        ("PASS LOST", 63.82, 20.55, 88.76, 22.55, None),
        ("PASS LOST", 85.60, 27.86, 94.41, 37.17, None),
        ("PASS LOST", 77.79, 27.53, 96.41, 25.37, None),
        ("PASS LOST", 91.09, 27.86, 109.54, 50.47, None),
        ("PASS LOST", 58.17, 26.04, 95.41, 40.33, None),
        ("PASS LOST", 53.35, 28.53, 73.80, 27.86, None),
        ("PASS LOST", 53.35, 34.02, 84.60, 58.62, None),
        ("PASS LOST", 56.18, 49.48, 97.07, 62.11, None),
        ("PASS LOST", 34.23, 74.91, 65.65, 78.57, None),
    ],
    "Nashville SC (03-28)": [
        ("PASS WON", 21.27, 14.23, 29.25, 31.02, None),
        ("PASS WON", 29.41, 23.38, 34.40, 64.60, None),
        ("PASS WON", 41.55, 39.67, 41.88, 6.92, None),
        ("PASS WON", 44.54, 32.52, 43.54, 14.23, None),
        ("PASS WON", 23.59, 56.46, 34.57, 47.48, None),
        ("PASS WON", 30.58, 64.44, 21.10, 49.48, None),
        ("PASS WON", 33.07, 56.79, 49.53, 69.59, None),
        ("PASS WON", 33.24, 59.78, 44.04, 71.75, None),
        ("PASS WON", 61.50, 71.58, 54.68, 75.57, None),
        ("PASS WON", 63.16, 50.81, 78.45, 67.26, None),
        ("PASS WON", 63.49, 76.90, 84.44, 62.77, None),
        ("PASS WON", 76.96, 56.96, 86.93, 57.79, None),
        ("PASS WON", 82.61, 59.12, 96.41, 68.43, None),
        ("PASS WON", 79.78, 35.35, 106.21, 11.74, None),
        ("PASS WON", 45.37, 49.64, 40.72, 32.02, None),
        ("PASS LOST", 78.62, 64.94, 96.57, 67.10, None),
        ("PASS LOST", 85.43, 68.76, 106.05, 77.74, None),
    ],
    "Seongnam FC (03-29)": [
        ("PASS WON", 28.08, 28.53, 29.75, 8.25, None),
        ("PASS WON", 33.74, 26.54, 29.41, 43.82, None),
        ("PASS WON", 28.08, 47.15, 31.57, 64.60, None),
        ("PASS WON", 39.39, 43.82, 51.69, 53.46, None),
        ("PASS WON", 43.88, 46.15, 55.84, 40.66, None),
        ("PASS WON", 47.03, 49.97, 44.04, 28.03, None),
        ("PASS WON", 47.53, 50.81, 71.97, 33.18, None),
        ("PASS WON", 67.65, 52.63, 64.32, 33.85, None),
        ("PASS WON", 73.63, 65.10, 69.31, 73.25, None),
        ("PASS WON", 77.29, 63.27, 79.12, 72.91, None),
        ("PASS WON", 81.61, 56.62, 93.91, 73.75, None),
        ("PASS WON", 86.43, 66.43, 81.78, 54.96, None),
        ("PASS WON", 111.03, 71.42, 99.56, 67.59, None),
        ("PASS WON", 89.76, 59.62, 97.74, 48.98, None),
        ("PASS WON", 88.43, 52.47, 96.41, 74.24, None),
        ("PASS WON", 87.93, 50.97, 77.12, 27.70, None),
        ("PASS WON", 81.61, 53.63, 74.30, 27.03, None),
        ("PASS WON", 79.28, 51.14, 94.91, 70.42, None),
        ("PASS WON", 52.85, 32.85, 65.49, 25.37, None),
        ("PASS WON", 82.77, 33.18, 69.31, 47.65, None),
        ("PASS LOST", 72.14, 16.56, 78.45, 1.60, None),
        ("PASS LOST", 79.62, 27.53, 97.07, 47.98, None),
        ("PASS LOST", 91.75, 50.14, 109.70, 65.77, None),
        ("PASS LOST", 96.41, 56.79, 107.04, 67.26, None),
    ],
    "NY Red Bulls (03-31)": [
        ("PASS WON", 39.39, 19.39, 52.35, 4.76, None),
        ("PASS WON", 63.82, 7.92, 72.63, 1.43, None),
        ("PASS WON", 70.47, 11.91, 80.95, 13.74, None),
        ("PASS WON", 64.49, 22.55, 97.24, 10.24, None),
        ("PASS WON", 32.07, 35.51, 43.04, 28.20, None),
        ("PASS WON", 53.52, 46.32, 54.02, 33.68, None),
        ("PASS WON", 77.12, 48.64, 84.94, 50.14, None),
        ("PASS WON", 78.12, 52.47, 117.52, 69.42, None),
        ("PASS WON", 88.76, 65.93, 97.40, 76.74, None),
        ("PASS WON", 82.61, 69.26, 86.60, 77.40, None),
        ("PASS WON", 78.62, 66.26, 79.62, 78.40, None),
        ("PASS WON", 83.61, 75.91, 62.49, 57.12, None),
        ("PASS WON", 34.40, 50.14, 88.76, 75.41, None),
        ("PASS WON", 56.68, 64.27, 78.29, 64.27, None),
        ("PASS WON", 51.85, 73.25, 54.18, 78.07, None),
        ("PASS WON", 41.05, 57.45, 46.04, 74.91, None),
        ("PASS WON", 37.39, 60.61, 41.71, 73.91, None),
        ("PASS WON", 30.41, 63.44, 36.89, 77.40, None),
        ("PASS WON", 26.09, 63.94, 28.42, 76.74, None),
        ("PASS WON", 22.43, 56.62, 22.10, 76.41, None),
        ("PASS WON", 33.90, 64.77, 25.42, 73.58, None),
        ("PASS LOST", 41.88, 42.49, 56.18, 52.97, None),
        ("PASS LOST", 37.56, 41.16, 46.37, 53.96, None),
        ("PASS LOST", 54.68, 56.96, 54.85, 64.44, None),
        ("PASS LOST", 51.69, 68.43, 66.15, 76.57, None),
    ],
}

DEFENSIVE_MATCHES_DATA = {
    "Michigan Wolves (02-20)": [
        ("DUEL_WON", 53.85, 25.21),
        ("DUEL_WON", 23.59, 29.69),
        ("DUEL_WON", 43.88, 50.31),
        ("DUEL_WON", 16.28, 50.47),
        ("DUEL_WON", 15.62, 72.08),
        ("DUEL_LOST", 73.63, 27.70),
        ("DUEL_LOST", 17.78, 75.41),
        ("INTERCEPTION", 65.82, 19.05),
        ("INTERCEPTION", 72.80, 57.62),
    ],
    "Philadelphia Union (02-27)": [
        ("DUEL_WON", 67.98, 34.68),
        ("DUEL_WON", 41.05, 23.54),
        ("DUEL_WON", 21.27, 31.36),
        ("DUEL_WON", 39.55, 60.95),
        ("DUEL_LOST", 29.08, 40.50),
        ("INTERCEPTION", 53.52, 19.05),
        ("INTERCEPTION", 28.08, 27.03),
        ("INTERCEPTION", 29.75, 53.63),
        ("INTERCEPTION", 30.58, 69.42),
        ("INTERCEPTION", 52.19, 58.12),
        ("INTERCEPTION", 59.17, 63.11),
        ("INTERCEPTION", 80.78, 68.92),
    ],
    "Columbus Crew (03-06)": [
        ("DUEL_WON", 22.76, 29.69),
        ("DUEL_WON", 48.36, 20.72),
        ("DUEL_LOST", 63.32, 56.96),
        ("DUEL_LOST", 25.42, 53.63),
        ("DUEL_LOST", 27.42, 34.35),
        ("DUEL_LOST", 35.06, 36.84),
        ("INTERCEPTION", 29.75, 35.84),
        ("INTERCEPTION", 29.41, 40.00),
        ("INTERCEPTION", 37.39, 60.95),
    ],
    "Minnesota United (03-13)": [
        ("DUEL_WON", 44.04, 58.95),
        ("DUEL_WON", 14.78, 18.56),
        ("DUEL_WON", 17.61, 12.24),
        ("DUEL_LOST", 77.29, 27.20),
        ("DUEL_LOST", 39.89, 3.43),
        ("DUEL_LOST", 33.24, 10.91),
        ("DUEL_LOST", 35.90, 57.12),
        ("DUEL_LOST", 0.99, 69.26),
        ("INTERCEPTION", 31.74, 38.50),
        ("INTERCEPTION", 35.06, 36.34),
        ("INTERCEPTION", 38.39, 41.00),
        ("INTERCEPTION", 46.54, 26.37),
        ("INTERCEPTION", 40.38, 19.22),
    ],
    "Vardar Soccer (03-14)": [
        ("INTERCEPTION", 72.63, 35.18),
        ("INTERCEPTION", 12.29, 44.99),
    ],
    "Colorado Rapids (03-20)": [
        ("DUEL_WON", 36.39, 73.75),
        ("DUEL_WON", 39.39, 68.76),
        ("DUEL_WON", 52.02, 66.10),
        ("DUEL_WON", 21.60, 53.63),
        ("DUEL_WON", 35.06, 43.32),
        ("DUEL_WON", 36.39, 31.36),
        ("DUEL_WON", 45.54, 25.04),
        ("DUEL_WON", 34.40, 21.71),
        ("DUEL_WON", 53.68, 17.23),
        ("DUEL_WON", 57.67, 22.55),
        ("DUEL_LOST", 78.95, 4.59),
        ("DUEL_LOST", 75.46, 65.43),
        ("DUEL_LOST", 33.07, 54.46),
        ("INTERCEPTION", 67.31, 9.58),
        ("INTERCEPTION", 39.89, 24.54),
        ("INTERCEPTION", 43.38, 28.86),
        ("INTERCEPTION", 27.92, 35.01),
        ("INTERCEPTION", 64.49, 53.80),
        ("INTERCEPTION", 36.56, 55.96),
        ("INTERCEPTION", 30.58, 62.11),
    ],
    "Connecticut United (03-27)": [
        ("DUEL_WON", 82.94, 3.43),
        ("DUEL_WON", 70.47, 21.05),
        ("DUEL_WON", 67.31, 27.53),
        ("DUEL_WON", 27.58, 32.52),
        ("DUEL_LOST", 65.49, 22.71),
        ("DUEL_LOST", 3.48, 72.42),
        ("INTERCEPTION", 82.28, 31.02),
        ("INTERCEPTION", 66.15, 26.04),
        ("INTERCEPTION", 83.94, 56.29),
        ("INTERCEPTION", 59.00, 61.44),
    ],
    "Nashville SC (03-28)": [
        ("DUEL_WON", 84.77, 54.79),
        ("DUEL_WON", 62.33, 55.46),
        ("DUEL_WON", 35.90, 62.61),
        ("DUEL_WON", 40.38, 70.09),
        ("DUEL_WON", 40.38, 40.33),
        ("DUEL_WON", 26.92, 23.71),
        ("DUEL_LOST", 92.91, 24.54),
        ("DUEL_LOST", 90.59, 53.63),
        ("DUEL_LOST", 64.82, 59.78),
        ("DUEL_LOST", 51.02, 71.58),
        ("INTERCEPTION", 85.60, 23.38),
        ("INTERCEPTION", 65.65, 57.12),
        ("INTERCEPTION", 77.45, 61.78),
    ],
    "Seongnam FC (03-29)": [
        ("DUEL_LOST", 73.80, 21.71),
        ("INTERCEPTION", 38.06, 30.36),
    ],
    "NY Red Bulls (03-31)": [
        ("DUEL_WON", 33.87, 59.39),
        ("DUEL_WON", 37.58, 67.14),
        ("DUEL_LOST", 66.32, 60.28),
        ("INTERCEPTION", 34.90, 34.02),
        ("INTERCEPTION", 56.34, 42.66),
        ("INTERCEPTION", 68.15, 54.30),
    ],
    "Minnesota United (04-10)": [
        ("DUEL_WON", 15.62, 54.30),
        ("DUEL_LOST", 36.39, 39.34),
        ("DUEL_LOST", 10.79, 64.27),
        ("INTERCEPTION", 66.15, 68.59),
        ("INTERCEPTION", 25.42, 54.79),
        ("INTERCEPTION", 35.06, 48.15),
        ("INTERCEPTION", 22.76, 21.88),
        ("INTERCEPTION", 56.84, 25.87),
        ("INTERCEPTION", 82.11, 20.72),
    ],
    "Sporting Kansas City (04-17)": [
        ("DUEL_WON", 85.43, 17.06),
        ("DUEL_WON", 76.12, 20.72),
        ("DUEL_WON", 54.68, 12.07),
        ("DUEL_WON", 53.18, 24.87),
        ("DUEL_WON", 24.92, 34.35),
        ("DUEL_WON", 31.24, 49.64),
        ("DUEL_WON", 39.05, 52.14),
        ("DUEL_WON", 43.71, 62.61),
        ("DUEL_WON", 49.69, 73.25),
        ("DUEL_WON", 75.79, 62.77),
        ("DUEL_LOST", 30.24, 69.09),
        ("INTERCEPTION", 60.83, 15.40),
        ("INTERCEPTION", 10.79, 25.87),
        ("INTERCEPTION", 52.35, 52.97),
        ("INTERCEPTION", 70.14, 61.28),
        ("INTERCEPTION", 54.85, 62.11),
        ("INTERCEPTION", 39.89, 66.60),
    ],
    "Cedar Stars (04-22)": [
        ("DUEL_WON", 9.30, 22.88),
        ("DUEL_WON", 59.00, 15.06),
        ("DUEL_WON", 60.83, 44.65),
        ("INTERCEPTION", 75.46, 28.20),
        ("INTERCEPTION", 79.95, 57.29),
        ("INTERCEPTION", 27.09, 66.43),
    ],
    "South Florida (04-23)": [
        ("DUEL_WON", 36.23, 32.85),
        ("DUEL_WON", 42.05, 54.79),
        ("DUEL_WON", 35.56, 57.62),
        ("DUEL_WON", 70.97, 18.72),
        ("INTERCEPTION", 55.18, 63.77),
        ("INTERCEPTION", 22.26, 62.94),
    ],
    "Real Salt Lake (04-26)": [
        ("DUEL_WON", 47.70, 56.96),
        ("DUEL_WON", 26.75, 55.29),
        ("DUEL_WON", 21.93, 26.37),
        ("DUEL_WON", 68.15, 2.93),
        ("DUEL_LOST", 76.29, 32.02),
        ("INTERCEPTION", 15.78, 53.30),
        ("INTERCEPTION", 35.23, 24.54),
        ("INTERCEPTION", 76.79, 21.55),
    ],
    "Real Futbol (05-23)": [
        ("DUEL_WON", 72.63, 10.24),
        ("DUEL_WON", 73.80, 13.90),
        ("DUEL_WON", 54.68, 40.50),
        ("DUEL_LOST", 69.97, 22.55),
        ("DUEL_LOST", 30.24, 5.26),
        ("DUEL_LOST", 39.22, 71.75),
        ("INTERCEPTION", 75.46, 56.12),
    ],
    "San Jose (05-24)": [
        ("DUEL_WON", 8.97, 23.21),
        ("DUEL_WON", 23.76, 23.71),
        ("DUEL_WON", 24.09, 41.50),
        ("DUEL_WON", 30.91, 61.61),
        ("DUEL_WON", 65.15, 39.17),
        ("DUEL_WON", 69.31, 29.36),
        ("DUEL_LOST", 27.42, 52.97),
        ("DUEL_LOST", 30.74, 49.48),
        ("DUEL_LOST", 34.73, 52.80),
        ("DUEL_LOST", 43.38, 59.62),
        ("DUEL_LOST", 34.90, 63.77),
        ("DUEL_LOST", 31.08, 62.61),
        ("DUEL_LOST", 21.27, 66.93),
        ("DUEL_LOST", 70.47, 57.79),
        ("INTERCEPTION", 76.62, 21.38),
        ("INTERCEPTION", 80.78, 60.61),
        ("INTERCEPTION", 21.93, 57.45),
        ("INTERCEPTION", 25.59, 70.59),
        ("INTERCEPTION", 34.90, 31.52),
        ("INTERCEPTION", 38.39, 33.68),
        ("INTERCEPTION", 29.91, 23.38),
    ],
    "Houston Dynamo (05-26)": [
        ("DUEL_WON", 68.31, 37.84),
        ("DUEL_WON", 68.15, 42.33),
        ("DUEL_WON", 83.27, 73.75),
        ("DUEL_WON", 55.51, 62.77),
        ("DUEL_WON", 49.53, 75.91),
        ("DUEL_WON", 31.24, 70.92),
        ("DUEL_WON", 24.59, 55.29),
        ("DUEL_LOST", 21.60, 21.88),
        ("DUEL_LOST", 26.59, 60.45),
    ],
}

# ── HELPERS ────────────────────────────────────────────────────
def apply_date_mapping(name: str) -> str:
    mapping = {
        "Connecticut United": "Connecticut United (03-27)",
        "Nashville SC": "Nashville SC (03-28)",
        "Seongnam FC": "Seongnam FC (03-29)",
        "NY Red Bulls": "NY Red Bulls (03-31)",
        "Real Salt Lake": "Real Salt Lake (04-26)",
        "Real Futbol": "Real Futbol (05-23)",
        "San Jose": "San Jose (05-24)",
        "Houston Dynamo": "Houston Dynamo (05-26)"
    }
    for k, v in mapping.items():
        if k.lower() == name.lower().strip():
            return v
    return name

def get_match_minutes(match_name: str) -> float:
    if match_name == "All Matches":
        total = 0.0
        for k in dfs_by_match:
            total += get_match_minutes(k)
        return total
    name_lower = match_name.lower()
    if "connecticut" in name_lower:
        return 60.0
    if "nashville" in name_lower:
        return 60.0
    if "seongnam" in name_lower:
        return 32.0
    if "red bulls" in name_lower:
        return 60.0
    if "houston" in name_lower:
        return 63.0
    if "vardar" in name_lower:
        return 65.0
    return 90.0

def read_docx_text(docx_path: Path) -> str:
    if not DOCX_AVAILABLE:
        raise RuntimeError("python-docx is not installed.")
    doc = Document(str(docx_path))
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())

def parse_docx_events(raw_text: str) -> dict:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    matches = {}
    current_match = None
    current_state = None
    re_match = re.compile(r"^Vs\s+(.+)$", re.IGNORECASE)
    re_success = re.compile(r"^Sucesso$", re.IGNORECASE)
    re_fail = re.compile(r"^Errado[s]?$", re.IGNORECASE)
    re_arrow = re.compile(
        r"^Seta\s+\d+:\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)\s*->\s*\(([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\)$",
        re.IGNORECASE,
    )
    for ln in lines:
        m_match = re_match.match(ln)
        if m_match:
            current_match = m_match.group(1).strip()
            matches.setdefault(current_match, [])
            current_state = None
            continue
        if re_success.match(ln):
            current_state = "PASS WON"
            continue
        if re_fail.match(ln):
            current_state = "PASS LOST"
            continue
        m_arrow = re_arrow.match(ln)
        if m_arrow and current_match and current_state:
            x1, y1, x2, y2 = map(float, m_arrow.groups())
            matches[current_match].append((current_state, x1, y1, x2, y2, None))
    return {k: v for k, v in matches.items() if len(v) > 0}

def load_docx_matches(docx_filename="Passes - Hudson Cicala.docx") -> dict:
    p = Path(docx_filename)
    if not p.exists():
        return {}
    txt = read_docx_text(p)
    return parse_docx_events(txt)

# DATA LOADING
docx_matches_data = {}
try:
    docx_matches_data = load_docx_matches()
except Exception:
    pass

combined_matches_data = {}
for k, v in docx_matches_data.items():
    mapped_k = apply_date_mapping(k)
    name = mapped_k if mapped_k not in combined_matches_data else f"DOCX - {mapped_k}"
    combined_matches_data[name] = v
for k, v in BASE_MATCHES_DATA.items():
    combined_matches_data[k] = v

if len(combined_matches_data) == 0:
    st.error("Could not load data.")
    st.stop()

# BUILD DATAFRAMES & REORDER MATCHES
dfs_by_match = {}
for match_name, events in combined_matches_data.items():
    dfm = pd.DataFrame(events, columns=["type", "x_start", "y_start", "x_end", "y_end", "video"])
    dfm["match"] = match_name
    dfm["number"] = np.arange(1, len(dfm) + 1)
    dfm["is_won"] = dfm["type"].str.contains("WON", case=False)
    dfm["progressive"] = dfm.apply(
        lambda r: r["is_won"] and is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]), axis=1
    )
    dfm["direction"] = dfm.apply(
        lambda r: classify_pass_direction(r["x_start"], r["y_start"], r["x_end"], r["y_end"]), axis=1
    )
    dfm["is_forward"] = dfm["direction"] == "forward"
    dfm["is_backward"] = dfm["direction"] == "backward"
    dfm["is_lateral"] = dfm["direction"].isin(["lateral_left", "lateral_right"])
    dfm["pass_distance"] = np.sqrt((dfm["x_end"] - dfm["x_start"]) ** 2 + (dfm["y_end"] - dfm["y_start"]) ** 2)
    dfm["xt_start"] = dfm.apply(lambda r: xt_value(r["x_start"], r["y_start"]), axis=1)
    dfm["xt_end"] = dfm.apply(lambda r: xt_value(r["x_end"], r["y_end"]), axis=1)
    dfm["delta_xt"] = np.where(dfm["is_won"], dfm["xt_end"] - dfm["xt_start"], 0.0)
    dfm["dist_bonus"] = distance_bonus(dfm["pass_distance"].values)
    dfm["delta_xt_adj"] = np.where(dfm["is_won"], dfm["delta_xt"] * (1.0 + dfm["dist_bonus"]), 0.0)
    dfs_by_match[match_name] = dfm

# REORDER LOGIC
items = list(dfs_by_match.items())
if len(items) >= 18:
    part1 = items[:6]
    part2 = items[14:18]
    part3 = items[6:14]
    part4 = items[18:]
    dfs_by_match = dict(part1 + part2 + part3 + part4)

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)

# DEFENSIVE DATA LOADING
defensive_dfs_by_match = {}
for match_name, events in DEFENSIVE_MATCHES_DATA.items():
    df_def = pd.DataFrame(events, columns=["type", "x", "y"])
    df_def["match"] = match_name
    df_def["is_attacking_half"] = df_def["x"] >= FIELD_X / 2
    df_def["is_duel_won"] = df_def["type"] == "DUEL_WON"
    df_def["is_duel_lost"] = df_def["type"] == "DUEL_LOST"
    df_def["is_duel"] = df_def["is_duel_won"] | df_def["is_duel_lost"]
    df_def["is_interception"] = df_def["type"] == "INTERCEPTION"
    df_def["in_funnel"] = df_def.apply(lambda r: is_in_funnel_zone(r["x"], r["y"]), axis=1)
    defensive_dfs_by_match[match_name] = df_def

# ── STATS ──────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame, match_name: str) -> dict:
    total = len(df)
    mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    if total == 0:
        return {
            "total_passes": 0, "successful_passes": 0, "unsuccessful_passes": 0,
            "accuracy_pct": 0.0, "progressive_attempted": 0, "progressive_successful": 0,
            "progressive_accuracy_pct": 0.0, "to_final_third_total": 0, "to_final_third_success": 0,
            "to_final_third_accuracy_pct": 0.0, "fwd": 0, "fwd_pct": 0.0, "bwd": 0, "bwd_pct": 0.0,
            "lat": 0, "lat_pct": 0.0, "pos_count": 0, "pos_pct": 0.0, "high_xt_pct": 0.0,
            "sum_dxt": 0.0, "total_p90": 0.0, "prog_p90": 0.0, "f3_p90": 0.0, "xt_p90": 0.0,
            "neg_xt_p90": 0.0, "minutes": mins, "long_acc_pct": 0.0, "high_xt_p90": 0.0, "dz_p90": 0.0,
            "advanced_passes_p90": 0.0, "advanced_accuracy_pct": 0.0,
        }
    successful = int(df["is_won"].sum())
    unsuccessful = total - successful
    accuracy = successful / total * 100.0
    progressive_total = int(df["progressive"].sum())
    progressive_unsuccessful = int(
        (~df["is_won"] & df.apply(
            lambda r: is_progressive_pass(r["x_start"], r["y_start"], r["x_end"], r["y_end"]), axis=1
        )).sum()
    )
    progressive_attempted = progressive_total + progressive_unsuccessful
    progressive_accuracy = (progressive_total / progressive_attempted * 100.0) if progressive_attempted else 0.0
    to_final_third = (df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)
    to_final_third_total = int(to_final_third.sum())
    to_final_third_success = int((to_final_third & df["is_won"]).sum())
    to_final_third_accuracy = (to_final_third_success / to_final_third_total * 100.0) if to_final_third_total else 0.0
    long_passes = df[df["pass_distance"] > 25.0]
    long_total = len(long_passes)
    long_success = int(long_passes["is_won"].sum())
    long_acc_pct = (long_success / long_total * 100.0) if long_total > 0 else 0.0
    dz_mask = df["is_won"] & (
        (df["x_end"] >= 100.0) | ((df["x_end"] >= 80.0) & (df["x_end"] < 100.0) & (df["y_end"] >= LANE_RIGHT_MAX) & (df["y_end"] < LANE_LEFT_MIN))
    )
    dz_passes = int(dz_mask.sum())
    fwd = int(df["is_forward"].sum())
    bwd = int(df["is_backward"].sum())
    lat = int(df["is_lateral"].sum())
    pos_count = int((df["is_won"] & (df["delta_xt_adj"] > 0)).sum())
    pos_pct = (pos_count / total * 100.0) if total > 0 else 0.0
    high_xt = int((df["delta_xt_adj"] > 0.1).sum())
    sum_dxt = float(df.loc[df["is_won"], "delta_xt_adj"].sum())
    neg_xt = float(df.loc[df["is_won"] & (df["delta_xt_adj"] < 0), "delta_xt_adj"].sum())
    advanced_successful = progressive_total + to_final_third_success
    advanced_attempted = progressive_attempted + to_final_third_total
    advanced_accuracy_pct = (advanced_successful / advanced_attempted * 100.0) if advanced_attempted else 0.0
    advanced_passes_p90 = round((progressive_total + to_final_third_success) * p90_factor, 2)
    return {
        "total_passes": total, "successful_passes": successful, "unsuccessful_passes": unsuccessful,
        "accuracy_pct": round(accuracy, 2), "progressive_attempted": progressive_attempted,
        "progressive_successful": progressive_total, "progressive_accuracy_pct": round(progressive_accuracy, 2),
        "to_final_third_total": to_final_third_total, "to_final_third_success": to_final_third_success,
        "to_final_third_accuracy_pct": round(to_final_third_accuracy, 2), "fwd": fwd, "fwd_pct": round(fwd / total * 100.0, 1),
        "bwd": bwd, "bwd_pct": round(bwd / total * 100.0, 1), "lat": lat, "lat_pct": round(lat / total * 100.0, 1),
        "pos_count": pos_count, "pos_pct": round(pos_pct, 1), "high_xt_pct": round(high_xt / total * 100.0, 1),
        "sum_dxt": round(sum_dxt, 3), "total_p90": round(total * p90_factor, 1), "prog_p90": round(progressive_total * p90_factor, 2),
        "f3_p90": round(to_final_third_success * p90_factor, 2), "xt_p90": round(sum_dxt * p90_factor, 3),
        "neg_xt_p90": round(neg_xt * p90_factor, 3), "minutes": mins, "long_acc_pct": round(long_acc_pct, 1),
        "high_xt_p90": round(high_xt * p90_factor, 2), "dz_p90": round(dz_passes * p90_factor, 2),
        "advanced_passes_p90": round(advanced_passes_p90, 1), "advanced_accuracy_pct": round(advanced_accuracy_pct, 2),
    }

def compute_defensive_stats(df: pd.DataFrame, match_name: str) -> dict:
    total_actions = len(df)
    if match_name == "All Matches":
        mins = sum(get_match_minutes(k) for k in defensive_dfs_by_match)
    else:
        mins = get_match_minutes(match_name)
    p90_factor = 90.0 / mins if mins > 0 else 1.0
    duels_won = int(df["is_duel_won"].sum())
    duels_lost = int(df["is_duel_lost"].sum())
    total_duels = duels_won + duels_lost
    duels_won_pct = (duels_won / total_duels * 100.0) if total_duels > 0 else 0.0
    interceptions = int(df["is_interception"].sum())
    attacking_half = df[df["is_attacking_half"]]
    actions_attacking = len(attacking_half)
    interceptions_attacking = int(attacking_half["is_interception"].sum())
    own_half = df[~df["is_attacking_half"]]
    actions_own = len(own_half)
    funnel_total = int(df["in_funnel"].sum())
    funnel_df = df[df["in_funnel"]]
    funnel_successful = int(funnel_df["is_duel_won"].sum() + funnel_df["is_interception"].sum())
    funnel_success_pct = (funnel_successful / funnel_total * 100.0) if funnel_total > 0 else 0.0
    return {
        "total_actions": total_actions, "total_actions_p90": round(total_actions * p90_factor, 1),
        "actions_own": actions_own, "actions_own_p90": round(actions_own * p90_factor, 1),
        "actions_attacking": actions_attacking, "actions_attacking_p90": round(actions_attacking * p90_factor, 1),
        "total_duels": total_duels, "duels_p90": round(total_duels * p90_factor, 1),
        "duels_won_pct": round(duels_won_pct, 1), "duels_won": duels_won,
        "interceptions": interceptions, "interceptions_p90": round(interceptions * p90_factor, 1),
        "interceptions_attacking": interceptions_attacking,
        "interceptions_attacking_p90": round(interceptions_attacking * p90_factor, 1),
        "funnel_actions": funnel_total, "funnel_actions_p90": round(funnel_total * p90_factor, 1),
        "funnel_success_pct": round(funnel_success_pct, 1),
    }

# ── UI HELPERS ─────────────────────────────────────────────────
# Card text sizes (labels/text only — numbers stay large)
CARD_TITLE_TEXT = "14px"
CARD_LABEL_TEXT = "16px"
CARD_SUBTEXT = "13px"
CARD_CAPTION = "12px"
CARD_BADGE_TEXT = "12px"

CARD_INNER_BORDER = "rgba(107,114,128,0.45)"
CARD_MUTED_TEXT = "#94a3b8"

def _lerp_channel(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * max(0.0, min(1.0, t))))

def _lerp_hex(hex_a: str, hex_b: str, t: float) -> str:
    ha, hb = hex_a.lstrip("#"), hex_b.lstrip("#")
    r = _lerp_channel(int(ha[0:2], 16), int(hb[0:2], 16), t)
    g = _lerp_channel(int(ha[2:4], 16), int(hb[2:4], 16), t)
    b = _lerp_channel(int(ha[4:6], 16), int(hb[4:6], 16), t)
    return f"#{r:02x}{g:02x}{b:02x}"

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

C_GREEN_LIGHT = "#86efac"
C_GREEN_STRONG = "#15803d"
C_ORANGE_LIGHT = "#fdba74"
C_RED_DARK = "#7f1d1d"
C_NEUTRAL = "#94a3b8"

def _diff_gradient_color(diff_pct: float) -> tuple:
    """Gradient color for diff % and status labels."""
    if abs(diff_pct) < 0.5:
        return C_NEUTRAL, "rgba(148,163,184,0.15)"
    if diff_pct > 0:
        t = min(diff_pct / 15.0, 1.0)
        color = _lerp_hex(C_GREEN_LIGHT, C_GREEN_STRONG, t)
        return color, _hex_to_rgba(color, 0.18)
    t = min(abs(diff_pct) / 15.0, 1.0)
    color = _lerp_hex(C_ORANGE_LIGHT, C_RED_DARK, t)
    return color, _hex_to_rgba(color, 0.18)

def _target_pct_diff(val: float, target: float) -> float:
    if target <= 0:
        return 0.0
    return ((val - target) / target) * 100.0

def _target_diff_badge_html(val: float, target: float) -> str:
    diff_pct = _target_pct_diff(val, target)
    if abs(diff_pct) < 0.5:
        text = "0%"
    elif diff_pct > 0:
        text = f"+{diff_pct:.0f}%"
    else:
        text = f"{diff_pct:.0f}%"
    color, bg = _diff_gradient_color(diff_pct)
    return (
        f'<span style="display:inline-block;padding:3px 9px;border-radius:7px;'
        f'font-size:{CARD_BADGE_TEXT};font-weight:700;color:{color};'
        f'background:{bg};border:1px solid {color}55">{text}</span>'
    )

def _target_delta_html(val: float, target: float) -> str:
    return _target_diff_badge_html(val, target)

def _accent_rgb(border_color):
    h = border_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

def _rand_target(base: float, key: str, is_pct: bool = False, decimals: int = 1) -> float:
    """Generate a stable pseudo-random target slightly above or below Hudson's baseline."""
    rng = np.random.default_rng(2026 + (hash(key) % 10000))
    sign = int(rng.choice([-1, 1]))
    if is_pct:
        target = float(np.clip(base + sign * rng.uniform(2.0, 8.5), 0.0, 100.0))
    elif base == 0:
        target = round(rng.uniform(0.5, 2.5), decimals)
        return target
    else:
        target = max(0.0, base + sign * base * rng.uniform(0.06, 0.16))
    return round(target, decimals)

BENCHMARK_POSITIONS = ("RDMF", "RCMF", "LDMF", "LCMF", "DMF")
BENCHMARK_EUR_KEY = "TOP 5 - EUR"
BENCHMARK_FILES = {"MLS": "MLS 1.xlsx", BENCHMARK_EUR_KEY: "TOP 5 - UE.xlsx"}
SGA_RANGE_METRICS = {
    "xt_p90": "0.8 – 2.0",
    "funnel_actions_p90": "2.0 – 5.0",
}
BENCHMARK_MINUTES_RATIO = 0.50

@st.cache_data(show_spinner=False)
def load_benchmark_targets(source: str) -> dict | None:
    """Mean per-90 benchmarks from Wyscout export for selected midfield positions."""
    filename = BENCHMARK_FILES.get(source)
    if not filename:
        return None
    path = Path(filename)
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path)
    except ImportError:
        return None
    if "Position" not in df.columns or "Minutes played" not in df.columns:
        return None
    df = df[df["Position"].isin(BENCHMARK_POSITIONS)].copy()
    max_mins = float(df["Minutes played"].max()) if len(df) else 0.0
    if max_mins <= 0:
        return None
    min_mins = max_mins * BENCHMARK_MINUTES_RATIO
    df = df[df["Minutes played"] >= min_mins]
    if df.empty:
        return None
    def_actions = (
        df["Defensive duels per 90"].fillna(0) + df["Interceptions per 90"].fillna(0)
    )
    prog_p90 = df["Progressive passes per 90"].fillna(0)
    f3_p90 = df["Passes to final third per 90"].fillna(0)
    prog_acc = df["Accurate progressive passes, %"].fillna(0)
    f3_acc = df["Accurate passes to final third, %"].fillna(0)
    prog_attempted_p90 = np.where(prog_acc > 0, prog_p90 / (prog_acc / 100.0), 0.0)
    f3_success_p90 = f3_p90 * (f3_acc / 100.0)
    advanced_passes_series = prog_p90 + f3_success_p90
    advanced_attempted_series = prog_attempted_p90 + f3_p90
    advanced_success_series = prog_p90 + f3_success_p90
    advanced_accuracy_series = np.where(
        advanced_attempted_series > 0,
        advanced_success_series / advanced_attempted_series * 100.0,
        0.0,
    )
    return {
        "total_p90": round(float(df["Passes per 90"].mean()), 1),
        "accuracy_pct": round(float(df["Accurate passes, %"].mean()), 1),
        "advanced_passes_p90": round(float(advanced_passes_series.mean()), 1),
        "advanced_accuracy_pct": round(float(advanced_accuracy_series.mean()), 1),
        "total_actions_p90": round(float(def_actions.mean()), 1),
        "duels_p90": round(float(df["Defensive duels per 90"].mean()), 1),
        "duels_won_pct": round(float(df["Defensive duels won, %"].mean()), 1),
        "sample_size": int(len(df)),
        "minutes_threshold": round(min_mins, 0),
    }

def build_metric_targets(pass_base: dict, def_base: dict, benchmark_source: str = "MLS") -> dict:
    bench = load_benchmark_targets(benchmark_source)
    targets = {
        "total_p90": bench["total_p90"] if bench else _rand_target(pass_base["total_p90"], "total_p90"),
        "accuracy_pct": bench["accuracy_pct"] if bench else _rand_target(pass_base["accuracy_pct"], "accuracy_pct", is_pct=True),
        "advanced_passes_p90": bench["advanced_passes_p90"] if bench else _rand_target(pass_base.get("advanced_passes_p90", 0), "advanced_passes_p90"),
        "advanced_accuracy_pct": bench["advanced_accuracy_pct"] if bench else _rand_target(pass_base.get("advanced_accuracy_pct", 0), "advanced_accuracy_pct", is_pct=True),
        "xt_p90": _rand_target(pass_base["xt_p90"], "xt_p90", decimals=2 if pass_base["xt_p90"] < 5 else 1),
        "pos_pct": _rand_target(pass_base["pos_pct"], "pos_pct", is_pct=True),
        "total_actions_p90": bench["total_actions_p90"] if bench else _rand_target(def_base["total_actions_p90"], "total_actions_p90"),
        "duels_p90": bench["duels_p90"] if bench else _rand_target(def_base["duels_p90"], "duels_p90"),
        "duels_won_pct": bench["duels_won_pct"] if bench else _rand_target(def_base["duels_won_pct"], "duels_won_pct", is_pct=True),
        "interceptions_p90": _rand_target(def_base["interceptions_p90"], "interceptions_p90"),
        "funnel_actions_p90": _rand_target(def_base["funnel_actions_p90"], "funnel_actions_p90"),
        "funnel_success_pct": _rand_target(def_base["funnel_success_pct"], "funnel_success_pct", is_pct=True),
    }
    targets["actions_own_p90"] = round(targets["total_actions_p90"] * 0.7, 1)
    return targets

def _fmt_target_value(key: str, targets: dict) -> str:
    v = targets[key]
    if key in (
        "accuracy_pct", "advanced_accuracy_pct", "pos_pct",
        "duels_won_pct", "funnel_success_pct",
    ):
        return f"{v:.1f}%"
    if key == "xt_p90":
        return f"{v:.1f}"
    return f"{v:.1f}"


def build_metric_item(label: str, val: float, disp_val: str, key: str, extra: str = ""):
    """Item tuple: label, val, disp_val, ref_type, ref_a, ref_b, extra."""
    if key in SGA_RANGE_METRICS:
        return (label, float(val), disp_val, "sga", SGA_RANGE_METRICS[key], "", extra)
    return (
        label,
        float(val),
        disp_val,
        "league",
        _fmt_target_value(key, T_MLS),
        _fmt_target_value(key, T_TOP_EUR),
        extra,
    )


def _sga_range_html(range_str: str) -> str:
    return (
        f'<div style="font-size:{CARD_CAPTION};color:{CARD_MUTED_TEXT};margin-top:6px">'
        f'Range SGA: {range_str}'
        f'</div>'
    )


def _targets_line_html(disp_mls: str, disp_top_eur: str, layout: str = "inline") -> str:
    if layout == "stacked":
        return (
            f'<div style="font-size:{CARD_CAPTION};color:{CARD_MUTED_TEXT};margin-top:6px;line-height:1.45">'
            f'<div>MLS: {disp_mls}</div>'
            f'<div>TOP 5 EUR: {disp_top_eur}</div>'
            f'</div>'
        )
    if layout == "columns":
        return (
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:6px">'
            f'<span style="font-size:{CARD_CAPTION};color:{CARD_MUTED_TEXT}">MLS: {disp_mls}</span>'
            f'<span style="font-size:{CARD_CAPTION};color:{CARD_MUTED_TEXT};text-align:right">TOP 5 EUR: {disp_top_eur}</span>'
            f'</div>'
        )
    return (
        f'<div style="font-size:{CARD_CAPTION};color:{CARD_MUTED_TEXT};margin-top:6px">'
        f'MLS: {disp_mls} · TOP 5 EUR: {disp_top_eur}'
        f'</div>'
    )


def _item_reference_line(item, layout: str = "inline") -> str:
    ref_type = item[3]
    if ref_type == "sga":
        return _sga_range_html(item[4])
    return _targets_line_html(item[4], item[5], layout=layout)


def _item_reference_text(item) -> str:
    if item[3] == "sga":
        return f"Range SGA: {item[4]}"
    return f"MLS: {item[4]} · TOP 5 EUR: {item[5]}"


def _combined_body_scoreboard(border_color, items, targets_layout="inline", tile=False):
    body = ""
    for idx, item in enumerate(items):
        label, disp_val = item[0], item[2]
        extra = item[6] if len(item) > 6 and item[6] else ""
        sep = _item_sep(idx, len(items))
        if tile:
            body += (
                f'<div style="{sep}background:rgba(0,0,0,0.18);border:1px solid {CARD_INNER_BORDER};'
                f'border-radius:12px;padding:12px 14px">'
            )
        else:
            body += f'<div style="{sep}">'
        body += (
            '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px">'
            f'<span style="font-size:{CARD_LABEL_TEXT};font-weight:600;color:#eef1f7;flex:1;min-width:0">{label}</span>'
            f'<span style="font-size:26px;font-weight:800;color:#ffffff;white-space:nowrap">{disp_val}</span>'
            '</div>'
        )
        body += _item_reference_line(item, layout=targets_layout)
        if extra:
            body += f'<div style="font-size:{CARD_CAPTION};color:#64748b;margin-top:4px;text-align:right">{extra}</div>'
        body += '</div>'
    return body


def _body_benchmarks_reference(items, targets_layout="inline"):
    body = ""
    for idx, item in enumerate(items):
        label = item[0]
        body += f'<div style="{_item_sep(idx, len(items))}">'
        body += f'<span style="font-size:{CARD_LABEL_TEXT};font-weight:600;color:#eef1f7">{label}</span>'
        body += _item_reference_line(item, layout=targets_layout)
        body += '</div>'
    return body


def _target_progress(val: float, target: float) -> float:
    if target <= 0:
        return 100.0 if val >= 0 else 0.0
    return float(np.clip((val / target) * 100.0, 0.0, 130.0))

def _kpi_status(val: float, target: float) -> tuple:
    """Return (status_key, label, color) — exceed, hit, close, miss, or miss_target vs target."""
    diff_pct = _target_pct_diff(val, target)
    color, _ = _diff_gradient_color(diff_pct)
    pct = _target_progress(val, target)
    if diff_pct < -10.0:
        return "miss_target", "Miss Target", color
    if val >= target:
        if diff_pct > 10.0:
            return "exceed", "Exceed Target", color
        return "hit", "Target Hit", color
    if pct >= 85.0:
        return "close", "Close to Target", color
    return "miss", "Below Target", color

def _kpi_icon(status_key: str) -> str:
    if status_key == "exceed":
        return "✓"
    if status_key == "hit":
        return "✓"
    if status_key == "close":
        return "~"
    if status_key == "miss_target":
        return "X"
    return "−"

def _metric_gradient_color(val: float, target: float) -> str:
    color, _ = _diff_gradient_color(_target_pct_diff(val, target))
    return color

def _item_sep(idx, total):
    return "" if idx == total - 1 else f"margin-bottom:14px;padding-bottom:14px;border-bottom:1px solid {CARD_INNER_BORDER};"

def _target_card_shell_html(title, border_color, body_html, compact=False):
    r, g, b = _accent_rgb(border_color)
    accent = f"rgb({r},{g},{b})"
    grad = (f"linear-gradient(150deg, rgba({r},{g},{b},0.18) 0%, "
            f"rgba(24,24,38,0.55) 55%, rgba(16,16,26,0.82) 100%)")
    pad = "14px 16px 12px 16px" if compact else "18px 20px 14px 20px"
    mb = "8px" if compact else "12px"
    html = (f'<div style="position:relative;background:{grad};'
            f'border:1px solid rgba({r},{g},{b},0.35);border-radius:16px;'
            f'padding:{pad};margin-bottom:{mb};'
            f'box-shadow:0 10px 28px rgba(0,0,0,0.40), inset 0 1px 0 rgba(255,255,255,0.06);'
            f'overflow:hidden">')
    html += (f'<div style="position:absolute;top:0;left:0;height:3px;width:100%;'
             f'background:linear-gradient(90deg, rgba({r},{g},{b},0.95), rgba({r},{g},{b},0.10))"></div>')
    html += '<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">'
    html += (f'<span style="width:8px;height:8px;border-radius:50%;background:{accent};'
             f'box-shadow:0 0 10px rgba({r},{g},{b},0.85)"></span>')
    html += (f'<span style="font-size:{CARD_TITLE_TEXT};font-weight:700;letter-spacing:1.1px;'
             f'text-transform:uppercase;color:#eef1f7">{title}</span>')
    html += '</div>'
    html += body_html
    html += '</div>'
    return html

def _target_card_shell(title, border_color, body_html, compact=False):
    st.markdown(_target_card_shell_html(title, border_color, body_html, compact=compact), unsafe_allow_html=True)

def _body_data_simple(items):
    body = ""
    for idx, item in enumerate(items):
        label, disp_val = item[0], item[2]
        extra = item[6] if len(item) > 6 and item[6] else ""
        body += f'<div style="{_item_sep(idx, len(items))}">'
        body += '<div style="display:flex;justify-content:space-between;align-items:center;gap:12px">'
        body += f'<span style="font-size:{CARD_LABEL_TEXT};font-weight:700;color:#eef1f7;flex:1;min-width:0">{label}</span>'
        body += f'<span style="font-size:26px;font-weight:800;color:#ffffff;white-space:nowrap">{disp_val}</span>'
        body += '</div>'
        if extra:
            body += f'<div style="font-size:{CARD_CAPTION};color:#64748b;margin-top:4px;text-align:right">{extra}</div>'
        body += '</div>'
    return body

def _body_target_row_header(label, right_html=""):
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:6px">'
        f'<span style="font-size:{CARD_LABEL_TEXT};font-weight:700;color:#eef1f7;flex:1;min-width:0">{label}</span>'
        f'{right_html}</div>'
    )

def _render_scoreboard_card(title, border_color, items, layout="combined", variant="inline", tile=False):
    if layout == "separated":
        _target_card_shell(title, border_color, _body_data_simple(items))
        _target_card_shell(
            f"{title} — Benchmarks",
            border_color,
            _body_benchmarks_reference(items, targets_layout=variant),
            compact=True,
        )
    else:
        _target_card_shell(
            title,
            border_color,
            _combined_body_scoreboard(border_color, items, targets_layout=variant, tile=tile),
        )


def _target_card_style_a(title, border_color, items, layout="combined"):
    _render_scoreboard_card(title, border_color, items, layout=layout, variant="inline")


def _target_card_style_b(title, border_color, items, layout="combined"):
    _render_scoreboard_card(title, border_color, items, layout=layout, variant="stacked")


def _target_card_style_c(title, border_color, items, layout="combined"):
    _render_scoreboard_card(title, border_color, items, layout=layout, variant="columns")


def _target_card_style_d(title, border_color, items, layout="combined"):
    _render_scoreboard_card(title, border_color, items, layout=layout, variant="inline", tile=False)


def _target_card_style_e(title, border_color, items, layout="combined"):
    _render_scoreboard_card(title, border_color, items, layout=layout, variant="stacked", tile=False)


def _target_card_style_f(title, border_color, items, layout="combined"):
    _render_scoreboard_card(title, border_color, items, layout=layout, variant="inline", tile=True)


TARGET_CARD_STYLES = {
    "A — Scoreboard (inline targets)": _target_card_style_a,
    "B — Scoreboard (stacked targets)": _target_card_style_b,
    "C — Scoreboard (column targets)": _target_card_style_c,
    "D — Scoreboard (compact)": _target_card_style_d,
    "E — Scoreboard (stacked)": _target_card_style_e,
    "F — Scoreboard (tiles)": _target_card_style_f,
}

CARD_EXPORT_WIDTH = 560
CARD_EXPORT_SCALE = 3
CARD_EXPORT_DPI = 300


def build_target_card_html(title, border_color, items, style_key, layout_mode="combined") -> str:
    variant_map = {
        "A — Scoreboard (inline targets)": ("inline", False),
        "B — Scoreboard (stacked targets)": ("stacked", False),
        "C — Scoreboard (column targets)": ("columns", False),
        "D — Scoreboard (compact)": ("inline", False),
        "E — Scoreboard (stacked)": ("stacked", False),
        "F — Scoreboard (tiles)": ("inline", True),
    }
    variant, tile = variant_map.get(style_key, ("inline", False))
    if layout_mode == "separated":
        inner = (
            _target_card_shell_html(title, border_color, _body_data_simple(items))
            + _target_card_shell_html(
                f"{title} — Benchmarks",
                border_color,
                _body_benchmarks_reference(items, targets_layout=variant),
                compact=True,
            )
        )
    else:
        inner = _target_card_shell_html(
            title,
            border_color,
            _combined_body_scoreboard(border_color, items, targets_layout=variant, tile=tile),
        )
    return (
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<style>'
        f'html,body{{margin:0;padding:24px;background:#0f0f1a;width:{CARD_EXPORT_WIDTH}px;'
        f'font-family:Arial,Helvetica,sans-serif;-webkit-font-smoothing:antialiased;}}'
        f'#card-export-root{{display:inline-block;width:100%;}}'
        f'</style></head><body><div id="card-export-root">{inner}</div></body></html>'
    )


def _find_chromium_executable() -> str | None:
    """Locate a headless Chromium/Chrome binary (Streamlit Cloud uses /usr/bin/chromium)."""
    candidates = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/local/bin/google-chrome",
    ]
    return next((p for p in candidates if os.path.isfile(p)), None)


def _screenshot_html_png(html: str) -> bytes | None:
    """Render dashboard card HTML to PNG (pixel-accurate). Playwright if available, else html2image."""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
        f.write(html)
        path = f.name
    try:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": CARD_EXPORT_WIDTH + 48, "height": 1400},
                    device_scale_factor=CARD_EXPORT_SCALE,
                )
                page.goto(f"file://{path}", wait_until="load")
                page.wait_for_timeout(150)
                png = page.locator("#card-export-root").screenshot(type="png")
                browser.close()
                return png
        except Exception:
            pass
        try:
            from html2image import Html2Image
            chrome_exe = _find_chromium_executable()
            if not chrome_exe:
                raise FileNotFoundError("No Chromium/Chrome binary found")
            user_dir = tempfile.mkdtemp()
            flags = [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                f"--user-data-dir={user_dir}",
                "--headless=new",
            ]
            hti = Html2Image(
                browser_executable=chrome_exe,
                custom_flags=flags,
                size=(CARD_EXPORT_WIDTH + 48, 1400),
            )
            out_dir = tempfile.mkdtemp()
            hti.output_path = out_dir
            hti.screenshot(html_file=path, save_as="card.png")
            out_path = os.path.join(out_dir, "card.png")
            if os.path.isfile(out_path):
                with open(out_path, "rb") as img_f:
                    return img_f.read()
        except Exception:
            pass
        return None
    finally:
        os.unlink(path)


def _export_card_png_matplotlib(title, border_color, items, layout_mode, dpi=CARD_EXPORT_DPI) -> bytes:
    from matplotlib.patches import FancyBboxPatch
    r, g, b = _accent_rgb(border_color)
    accent = (r / 255.0, g / 255.0, b / 255.0)
    rows = []
    if layout_mode == "separated":
        for item in items:
            rows.append((item[0], str(item[2]), "", "#eef1f7"))
        for item in items:
            rows.append((item[0], "", _item_reference_text(item), CARD_MUTED_TEXT))
    else:
        for item in items:
            rows.append((item[0], str(item[2]), _item_reference_text(item), CARD_MUTED_TEXT))
    fig_w = (CARD_EXPORT_WIDTH / CARD_EXPORT_DPI) * CARD_EXPORT_SCALE
    fig_h = (1.35 + len(rows) * 0.58) * CARD_EXPORT_SCALE
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=CARD_EXPORT_DPI)
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card = FancyBboxPatch(
        (0.03, 0.04), 0.94, 0.92, boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2, edgecolor=accent, facecolor=(0.10, 0.10, 0.16, 1.0),
    )
    ax.add_patch(card)
    ax.plot([0.05, 0.95], [0.90, 0.90], color=accent, linewidth=2.5)
    ax.text(0.06, 0.94, title.upper(), color="#eef1f7", fontsize=13, fontweight="bold", va="top")
    y = 0.84
    for label, value, subline, color in rows:
        ax.text(0.07, y, label, color="#c7cdda", fontsize=10, fontweight="bold", va="top")
        if value:
            ax.text(0.93, y, value, color="#ffffff", fontsize=16, fontweight="bold", ha="right", va="top")
        if subline:
            ax.text(0.07, y - 0.055, subline, color=color, fontsize=9, va="top")
            y -= 0.13
        else:
            y -= 0.09
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor="#0f0f1a", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_target_card_png(
    title, border_color, items, style_key, layout_mode="combined",
) -> bytes:
    html = build_target_card_html(title, border_color, items, style_key, layout_mode)
    png = _screenshot_html_png(html)
    if png:
        return png
    return _export_card_png_matplotlib(title, border_color, items, layout_mode, dpi=CARD_EXPORT_DPI)


def target_section_card(title, border_color, items, style_key, layout_mode="combined", export_key=None):
    renderer = TARGET_CARD_STYLES.get(style_key, _target_card_style_a)
    renderer(title, border_color, items, layout=layout_mode)
    if export_key:
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", title.strip()) or "card"
        png_bytes = export_target_card_png(title, border_color, items, style_key, layout_mode)
        st.download_button(
            label="Export card PNG (HD)",
            data=png_bytes,
            file_name=f"Hudson_{safe_name}.png",
            mime="image/png",
            key=export_key,
            use_container_width=True,
        )

# ── DRAW HELPERS (PITCH) ───────────────────────────────────────
def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg); fig.set_dpi(FIG_DPI)
    ax.axvline(x=FINAL_THIRD_LINE_X, color="#ffffff", lw=1.2, alpha=0.40, linestyle="--")
    ax.axvline(x=HALF_LINE_X, color="#ffffff", lw=0.7, alpha=0.12, linestyle="--")
    return fig, ax, pitch

def _attack_arrow(fig, has_cbar=False):
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(FancyArrowPatch((0.44 + ox, 0.045), (0.56 + ox, 0.045), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=11, linewidth=1.6, color="#aaaaaa"))
    fig.text(0.50 + ox, 0.012, "Attacking Direction", ha="center", va="bottom", transform=fig.transFigure,
             fontsize=7.5, color="#aaaaaa")

def _save_fig(fig):
    fig.canvas.draw(); buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0); return Image.open(buf)

def draw_pass_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        is_lost = not row["is_won"]; is_prog = bool(row["progressive"])
        if is_lost:
            color, alpha = COLOR_FAIL, 0.72
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, 0.88
        else:
            color, alpha = COLOR_SUCCESS, ALPHA_SUCCESS
        pitch.arrows(row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                     color=color, width=1.3, headwidth=2.0, headlength=2.0, ax=ax, zorder=3, alpha=alpha)
        pitch.scatter(row["x_start"], row["y_start"], s=32, marker="o", color=color,
                      edgecolors="white", linewidths=0.6, ax=ax, zorder=6, alpha=alpha)
    leg = ax.legend(handles=[
        Line2D([0],[0],color=COLOR_SUCCESS,lw=2.0,label="Completed",alpha=0.65),
        Line2D([0],[0],color=COLOR_PROGRESSIVE,lw=2.0,label="Progressive",alpha=0.90),
        Line2D([0],[0],color=COLOR_FAIL,lw=2.0,label="Incomplete",alpha=0.90)
    ],loc="upper left",bbox_to_anchor=(0.01,0.99),frameon=True,facecolor="#1a1a2e",
       edgecolor="#444466",fontsize=6.5,labelspacing=0.35,borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90); _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_corridor_heatmap(df):
    df_s = df[df["is_won"]].copy()
    x_bins = np.linspace(0.0, FIELD_X, 7)
    corridors = {"left": (LANE_LEFT_MIN, FIELD_Y), "center": (LANE_RIGHT_MAX, LANE_LEFT_MIN), "right": (0.0, LANE_RIGHT_MAX)}
    counts = {}
    for cname, (y0, y1) in corridors.items():
        arr = np.zeros(6, dtype=int)
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]
            arr[i] = int(((df_s["x_end"] >= x0_) & (df_s["x_end"] < x1_) & (df_s["y_end"] >= y0) & (df_s["y_end"] < y1)).sum())
        counts[cname] = arr
    all_vals = np.concatenate([counts[c] for c in counts]); vmax = max(1, int(all_vals.max()))
    cmap = LinearSegmentedColormap.from_list("wr", ["#ffffff", "#ffecec", "#ffbfbf", "#ff8080", "#ff3b3b", "#ff0000"])
    norm = Normalize(vmin=0, vmax=vmax); threshold = max(1, vmax * 0.35)
    fig, ax, pitch = _base_pitch()
    for cname, (y0, y1) in corridors.items():
        for i in range(6):
            x0_, x1_ = x_bins[i], x_bins[i + 1]; value = counts[cname][i]
            ax.add_patch(Rectangle((x0_, y0), x1_ - x0_, y1 - y0,
                                   facecolor=cmap(norm(value)), edgecolor=(1,1,1,0.12), lw=0.5, alpha=0.95, zorder=2))
            ax.text((x0_+x1_)/2, (y0+y1)/2, str(value), ha="center", va="center",
                    color="#000000" if value <= threshold else "#ffffff",
                    fontsize=9, fontweight="700" if value>=vmax*0.5 else "600", zorder=4)
    ax.axhline(y=LANE_LEFT_MIN, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX, color="#ffffff", lw=0.5, alpha=0.15, linestyle="--", zorder=3)
    _attack_arrow(fig); return _save_fig(fig), fig

def _draw_comet_arrow(ax, x0, y0, x1, y1, color):
    segs = 12; ts = np.linspace(0.0, 1.0, segs + 1)
    for i in range(segs):
        t0, t1 = ts[i], ts[i+1]; xa=x0+(x1-x0)*t0; ya=y0+(y1-y0)*t0; xb=x0+(x1-x0)*t1; yb=y0+(y1-y0)*t1
        alpha = 0.85*(0.15+0.85*t1); lw = 2.5*(0.80+0.20*t1)
        ax.plot([xa,xb],[ya,yb],color=color,linewidth=lw,alpha=alpha,zorder=4,solid_capstyle="round")
    ax.scatter(x0,y0,s=20,marker="o",facecolors="none",edgecolors=color,linewidths=1.5,zorder=5,alpha=0.85)
    ax.scatter(x1,y1,s=32,marker="o",facecolors=color,edgecolors="white",linewidths=0.9,zorder=6,alpha=0.85)

def draw_top_xt_map(df, top_n=5):
    fig, ax, pitch = _base_pitch()
    top_passes = (df[(df["is_won"])&(df["delta_xt_adj"]>0)].sort_values("delta_xt_adj",ascending=False).head(top_n).copy().reset_index(drop=True))
    cursor_points = []
    if not top_passes.empty:
        for _, row in top_passes.iterrows():
            val = float(row["delta_xt_adj"]); color = CMAP_TOP10(NORM_TOP10(np.clip(val,0.05,0.40)))
            _draw_comet_arrow(ax,float(row["x_start"]),float(row["y_start"]),float(row["x_end"]),float(row["y_end"]),color)
            match_name = row.get("match", "")
            pt = ax.scatter(float(row["x_start"]), float(row["y_start"]), s=20, marker="o",
                           facecolors="none", edgecolors=color, linewidths=1.5, zorder=5, alpha=0, visible=False)
            cursor_points.append((pt, f"xT: {val:.3f}\nMatch: {match_name}"))
        crs = mplcursors.cursor([p[0] for p in cursor_points], hover=True)
        @crs.connect("add")
        def _(sel):
            sel.annotation.set_text(cursor_points[sel.index][1])
            sel.annotation.get_bbox_patch().set(fc="#1a1a2e", ec="#5b9bd5", alpha=0.95)
            sel.annotation.arrow_patch.set(connectionstyle="arc3,rad=0.2", fc="#1a1a2e", ec="#5b9bd5")
    sm = plt.cm.ScalarMappable(cmap=CMAP_TOP10,norm=NORM_TOP10)
    cbar=fig.colorbar(sm,ax=ax,fraction=0.020,pad=0.02,shrink=0.60); cbar.set_label("Pass Impact",color="#ffffff",fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#ffffff",labelsize=7); plt.setp(plt.getp(cbar.ax.axes,"yticklabels"),color="#ffffff")
    _attack_arrow(fig,has_cbar=True); return _save_fig(fig), fig

# ── DEFENSIVE PITCH DRAW HELPERS ───────────────────────────────
COLOR_DUEL_WON="#10b981"; COLOR_DUEL_LOST="#E07070"; COLOR_INTERCEPTION="#2F80ED"

def draw_defensive_map(df):
    fig, ax, pitch = _base_pitch()
    for _, row in df.iterrows():
        if row["is_duel_won"]:
            color,marker,s,alpha=COLOR_DUEL_WON,"o",90,0.85
        elif row["is_duel_lost"]:
            color,marker,s,alpha=COLOR_DUEL_LOST,"X",100,0.85
        else:
            color,marker,s,alpha=COLOR_INTERCEPTION,"^",80,0.85
        pitch.scatter(row["x"],row["y"],s=s,marker=marker,color=color,edgecolors="white",linewidths=0.8,ax=ax,zorder=6,alpha=alpha)
    leg=ax.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor=COLOR_DUEL_WON,markersize=7,label="Duel Won",alpha=0.90),
        Line2D([0],[0],marker="X",color="w",markerfacecolor=COLOR_DUEL_LOST,markersize=8,label="Duel Lost",alpha=0.90),
        Line2D([0],[0],marker="^",color="w",markerfacecolor=COLOR_INTERCEPTION,markersize=7,label="Interception",alpha=0.90)
    ],loc="upper left",bbox_to_anchor=(0.01,0.99),frameon=True,facecolor="#1a1a2e",edgecolor="#444466",fontsize=6.5,labelspacing=0.35,borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90); _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_funnel_protection_map(df):
    fig, ax, pitch = _base_pitch()
    funnel_rect=Rectangle((0,PENALTY_AREA_Y_MIN),FUNNEL_X_EXTEND,PENALTY_AREA_Y_MAX-PENALTY_AREA_Y_MIN,
                          facecolor="#ffd700",edgecolor="#ffd700",lw=1.5,linestyle="--",alpha=0.12,zorder=2)
    ax.add_patch(funnel_rect)
    cursor_points = []
    for _, row in df.iterrows():
        x,y=float(row["x"]),float(row["y"]); in_funnel=bool(row.get("in_funnel",is_in_funnel_zone(x,y)))
        match_name = row.get("match", "")
        action_type = row["type"]
        if in_funnel:
            marker,s,color,edge="*",120,"#ffd700","#b8860b"
            pt = ax.scatter(x, y, s=1, marker="o", color="none", edgecolors="none", linewidths=0, zorder=1, alpha=0, visible=False)
            cursor_points.append((pt, f"{action_type}\nMatch: {match_name}"))
        else:
            marker,s,color,edge="o",60,"#888888","#555555"
        pitch.scatter(x,y,s=s,marker=marker,color=color,edgecolors=edge,linewidths=0.5,ax=ax,zorder=6,alpha=0.85)
    if cursor_points:
        crs = mplcursors.cursor([p[0] for p in cursor_points], hover=True)
        @crs.connect("add")
        def _(sel):
            sel.annotation.set_text(cursor_points[sel.index][1])
            sel.annotation.get_bbox_patch().set(fc="#1a1a2e", ec="#ffd700", alpha=0.95)
            sel.annotation.arrow_patch.set(connectionstyle="arc3,rad=0.2", fc="#1a1a2e", ec="#ffd700")
    leg=ax.legend(handles=[
        Line2D([0],[0],marker="*",color="w",markerfacecolor="#ffd700",markersize=9,label="Funnel Action",alpha=0.95),
        Line2D([0],[0],marker="o",color="w",markerfacecolor="#888888",markersize=6,label="Other Action",alpha=0.50)
    ],loc="upper left",bbox_to_anchor=(0.01,0.99),frameon=True,facecolor="#1a1a2e",edgecolor="#444466",fontsize=6.5,labelspacing=0.35,borderpad=0.4)
    for t in leg.get_texts(): t.set_color("white")
    leg.get_frame().set_alpha(0.90); _attack_arrow(fig)
    return _save_fig(fig), fig

def draw_defensive_heatmap(df):
    corridors={"Right":(LANE_LEFT_MIN,FIELD_Y),"Center":(LANE_RIGHT_MAX,LANE_LEFT_MIN),"Left":(0.0,LANE_RIGHT_MAX)}
    corridor_data={}
    for cname,(y0,y1) in corridors.items():
        mask=(df["y"]>=y0)&(df["y"]<y1); val=int(mask.sum())
        duel_mask=mask&df["is_duel"]; duels_won=int((df.loc[duel_mask,"is_duel_won"]).sum()); duels_lost=int((df.loc[duel_mask,"is_duel_lost"]).sum())
        duel_pct=(duels_won/(duels_won+duels_lost)*100.0) if (duels_won+duels_lost)>0 else None
        corridor_data[cname]={"total":val,"duels_won":duels_won,"duels_lost":duels_lost,"duel_pct":duel_pct}
    all_vals=[corridor_data[c]["total"] for c in corridors]; vmax=max(1,max(all_vals))
    cmap=LinearSegmentedColormap.from_list("def_cm_blue",["#6ab0f5","#3a8ad0","#2a5a8a","#1a3058","#0a1428"],N=20)
    norm=Normalize(vmin=0,vmax=max(vmax,1)); threshold=max(1,vmax*0.40)
    fig,ax,pitch=_base_pitch()
    for cname,(y0,y1) in corridors.items():
        c=corridor_data[cname]; value=c["total"]
        rect=Rectangle((0,y0),FIELD_X,y1-y0,facecolor=cmap(norm(value)),edgecolor=(1,1,1,0.08),lw=0.5,alpha=0.95,zorder=2)
        ax.add_patch(rect)
        duel_pct = c['duel_pct']
        if duel_pct is not None:
            label=f"{cname}\nTotal: {value}\nWon: {c['duels_won']}/{c['duels_won']+c['duels_lost']} ({duel_pct:.0f}%)"
        else:
            label=f"{cname}\nTotal: {value}"
        ax.text(FIELD_X/2,(y0+y1)/2,label,ha="center",va="center",
                color="#000000" if value<=threshold else "#ffffff",fontsize=9,fontweight="600",zorder=4)
    ax.axhline(y=LANE_LEFT_MIN,color="#ffffff",lw=0.5,alpha=0.20,linestyle="--",zorder=3)
    ax.axhline(y=LANE_RIGHT_MAX,color="#ffffff",lw=0.5,alpha=0.20,linestyle="--",zorder=3)
    _attack_arrow(fig); return _save_fig(fig), fig

# ── PDF REPORT ─────────────────────────────────────────────────
PDF_BG = "#0f0f1a"
PDF_CARD_BG = "#1a1a2e"
PDF_TEXT = "#eef1f7"
PDF_MUTED = "#8b93a7"
PDF_LABEL = "#c7cdda"

def _fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return buf

def _pil_to_png_bytes(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def _pdf_draw_dark_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(rl_colors.HexColor(PDF_BG))
    canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1, stroke=0)
    canvas.restoreState()

def _pdf_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "PdfTitle", parent=styles["Heading1"],
            fontSize=20, textColor=rl_colors.HexColor(PDF_TEXT),
            spaceAfter=4, alignment=1, fontName="Helvetica-Bold",
        ),
        "sub": ParagraphStyle(
            "PdfSub", parent=styles["Normal"],
            fontSize=9, textColor=rl_colors.HexColor(PDF_MUTED), alignment=1,
        ),
        "section": ParagraphStyle(
            "PdfSection", parent=styles["Heading2"],
            fontSize=13, textColor=rl_colors.HexColor(PASS_TONES[1]),
            spaceBefore=0, spaceAfter=6, fontName="Helvetica-Bold",
        ),
        "map_label": ParagraphStyle(
            "PdfMapLabel", parent=styles["Normal"],
            fontSize=10, textColor=rl_colors.HexColor("#cccccc"),
            spaceAfter=3, fontName="Helvetica-Bold",
        ),
        "card_title": ParagraphStyle(
            "PdfCardTitle", parent=styles["Normal"],
            fontSize=9, textColor=rl_colors.white, fontName="Helvetica-Bold",
            leading=11,
        ),
        "metric_label": ParagraphStyle(
            "PdfMetricLabel", parent=styles["Normal"],
            fontSize=9, textColor=rl_colors.HexColor(PDF_LABEL), leading=11,
        ),
        "metric_tgt": ParagraphStyle(
            "PdfMetricTgt", parent=styles["Normal"],
            fontSize=8, textColor=rl_colors.HexColor(PDF_MUTED), leading=10,
        ),
    }

def _pdf_status_text(val, target):
    diff_pct = _target_pct_diff(val, target)
    color, _ = _diff_gradient_color(diff_pct)
    pct = _target_progress(val, target)
    if diff_pct < -10.0:
        return "Miss Target", color
    if val >= target:
        if diff_pct > 10.0:
            return "Exceed Target", color
        return "Target Hit", color
    if pct >= 85.0:
        return "Close to Target", color
    return "Below Target", color

def _pdf_usable_width():
    return landscape(A4)[0] - 2.0 * cm

def _pdf_col_width(n_cols: int = 3, gap_cm: float = 0.15):
    return (_pdf_usable_width() - gap_cm * cm * (n_cols - 1)) / n_cols

def _pdf_rl_image(img_buf, target_width):
    img_buf.seek(0)
    with Image.open(img_buf) as pil_img:
        iw, ih = pil_img.size
    img_buf.seek(0)
    w = target_width
    h = w * (ih / max(iw, 1))
    return RLImage(img_buf, width=w, height=h), h

def _pdf_diff_badge_text(val, target):
    diff_pct = _target_pct_diff(val, target)
    if abs(diff_pct) < 0.5:
        text = "0%"
    elif diff_pct > 0:
        text = f"+{diff_pct:.0f}%"
    else:
        text = f"{diff_pct:.0f}%"
    color, _ = _diff_gradient_color(diff_pct)
    return text, color

def _pdf_reference_text(ref_type: str, ref_a: str, ref_b: str = "") -> str:
    if ref_type == "sga":
        return f"Range SGA: {ref_a}"
    return f"MLS: {ref_a} · TOP 5 EUR: {ref_b}"


def _pdf_dark_card(accent_hex, title, metrics, pstyles, card_width):
    """metrics: list of (label, disp_val, ref_type, ref_a, ref_b='')"""
    label_w = card_width * 0.56
    val_w = card_width * 0.44
    rows = [[Paragraph(title.upper(), pstyles["card_title"]), ""]]
    row_idx = 0
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor(PDF_CARD_BG)),
        ("BOX", (0, 0), (-1, -1), 0.8, rl_colors.HexColor(accent_hex)),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(accent_hex)),
        ("SPAN", (0, 0), (-1, 0)),
    ]
    for label, disp_val, ref_type, ref_a, *rest in metrics:
        ref_b = rest[0] if rest else ""
        row_idx += 1
        cell = [
            Paragraph(f'<b>{label}</b>', pstyles["metric_label"]),
            Paragraph(
                f'<font color="{PDF_TEXT}"><b>{disp_val}</b></font><br/>'
                f'<font color="{PDF_MUTED}">{_pdf_reference_text(ref_type, ref_a, ref_b)}</font>',
                pstyles["metric_tgt"],
            ),
        ]
        rows.append(cell)
        style_cmds.append(("LINEABOVE", (0, row_idx), (-1, row_idx), 0.3, rl_colors.HexColor("#6b7280")))
    tbl = Table(rows, colWidths=[label_w, val_w])
    tbl.setStyle(TableStyle(style_cmds))
    return tbl

def _pdf_map_cell(label, img_buf, pstyles, col_width):
    rl_img, _ = _pdf_rl_image(img_buf, col_width)
    cell = Table([
        [Paragraph(label, pstyles["map_label"])],
        [rl_img],
    ], colWidths=[col_width])
    cell.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor(PDF_BG)),
    ]))
    return cell

def _pdf_dashboard_section(section_title, map_entries, stat_cards, pstyles):
    col_w = _pdf_col_width(3)
    map_cells = [_pdf_map_cell(label, buf, pstyles, col_w) for label, buf in map_entries]
    maps_row = Table([map_cells], colWidths=[col_w] * 3, hAlign="LEFT")
    maps_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-2, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor(PDF_BG)),
    ]))
    cards_row = Table([stat_cards], colWidths=[col_w] * 3, hAlign="LEFT")
    cards_row.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-2, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor(PDF_BG)),
    ]))
    return [
        Paragraph(section_title, pstyles["section"]),
        maps_row,
        Spacer(1, 0.12 * cm),
        cards_row,
    ]

def generate_season_pdf(card_tones=None, benchmark_source="MLS"):
    if not PDF_AVAILABLE:
        raise RuntimeError("reportlab is not installed. Run: pip install reportlab")
    tones = card_tones or PASS_TONES

    df_pass = pd.concat(dfs_by_match.values(), ignore_index=True)
    df_def = pd.concat(defensive_dfs_by_match.values(), ignore_index=True)

    pass_stats_list = [compute_stats(dfs_by_match[m], m) for m in dfs_by_match]
    s_pass = {}
    for k in pass_stats_list[0].keys():
        if isinstance(pass_stats_list[0][k], (int, float)):
            s_pass[k] = sum(s[k] for s in pass_stats_list) / len(pass_stats_list)
        else:
            s_pass[k] = 0

    def_all = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]
    d_def = {}
    for k in def_all[0].keys():
        if isinstance(def_all[0][k], (int, float)):
            d_def[k] = sum(s[k] for s in def_all) / len(def_all)
        else:
            d_def[k] = 0

    _pb, _db = {}, {}
    if pass_stats_list:
        for k in pass_stats_list[0].keys():
            if isinstance(pass_stats_list[0][k], (int, float)):
                _pb[k] = sum(s[k] for s in pass_stats_list) / len(pass_stats_list)
    if def_all:
        for k in def_all[0].keys():
            if isinstance(def_all[0][k], (int, float)):
                _db[k] = sum(s[k] for s in def_all) / len(def_all)
    T_pdf_mls = build_metric_targets(_pb, _db, "MLS")
    T_pdf_eur = build_metric_targets(_pb, _db, BENCHMARK_EUR_KEY)

    img_pm, fig_pm = draw_pass_map(df_pass); plt.close(fig_pm)
    img_ht, fig_ht = draw_corridor_heatmap(df_pass); plt.close(fig_ht)
    _, fig_xt = draw_top_xt_map(df_pass, top_n=10)
    buf_xt = _fig_to_png_bytes(fig_xt); plt.close(fig_xt)
    img_def, fig_def = draw_defensive_map(df_def); plt.close(fig_def)
    img_dhm, fig_dhm = draw_defensive_heatmap(df_def); plt.close(fig_dhm)
    img_fun, fig_fun = draw_funnel_protection_map(df_def); plt.close(fig_fun)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=landscape(A4),
        leftMargin=1.0 * cm, rightMargin=1.0 * cm,
        topMargin=0.8 * cm, bottomMargin=0.8 * cm,
    )
    ps = _pdf_styles()
    col_w = _pdf_col_width(3)

    pass_maps = [
        ("Pass Map", _pil_to_png_bytes(img_pm)),
        ("Zone Heatmap (Destination)", _pil_to_png_bytes(img_ht)),
        ("Top 10 Pass Impact", buf_xt),
    ]
    pass_cards = [
        _pdf_dark_card(tones[0], "Overview", [
            ("Total Passes Per Game", f"{s_pass['total_p90']:.1f}", "league", f"{T_pdf_mls['total_p90']:.1f}", f"{T_pdf_eur['total_p90']:.1f}"),
            ("% Accuracy", f"{s_pass['accuracy_pct']:.1f}%", "league", f"{T_pdf_mls['accuracy_pct']:.1f}%", f"{T_pdf_eur['accuracy_pct']:.1f}%"),
        ], ps, col_w),
        _pdf_dark_card(tones[1], "Progressive", [
            ("Progressive Passes Per Game", f"{s_pass['advanced_passes_p90']:.1f}", "league", f"{T_pdf_mls['advanced_passes_p90']:.1f}", f"{T_pdf_eur['advanced_passes_p90']:.1f}"),
            ("% Progressive Accuracy", f"{s_pass['advanced_accuracy_pct']:.1f}%", "league", f"{T_pdf_mls['advanced_accuracy_pct']:.1f}%", f"{T_pdf_eur['advanced_accuracy_pct']:.1f}%"),
        ], ps, col_w),
        _pdf_dark_card(tones[2], "Impact", [
            ("Pass Impact Value Per Game", f"{s_pass['xt_p90']:.1f}", "sga", SGA_RANGE_METRICS["xt_p90"]),
            ("% Positive Impact", f"{s_pass['pos_pct']:.1f}%", "league", f"{T_pdf_mls['pos_pct']:.1f}%", f"{T_pdf_eur['pos_pct']:.1f}%"),
        ], ps, col_w),
    ]

    def_maps = [
        ("Defensive Actions Map", _pil_to_png_bytes(img_def)),
        ("Defensive Heatmap", _pil_to_png_bytes(img_dhm)),
        ("Funnel Protection Actions", _pil_to_png_bytes(img_fun)),
    ]
    def_cards = [
        _pdf_dark_card(tones[0], "Overview", [
            ("Defensive Actions Per Game", f"{d_def['total_actions_p90']:.1f}", "league", f"{T_pdf_mls['total_actions_p90']:.1f}", f"{T_pdf_eur['total_actions_p90']:.1f}"),
            ("Actions in Own Half Per Game", f"{d_def['actions_own_p90']:.1f}", "league", f"{T_pdf_mls['actions_own_p90']:.1f}", f"{T_pdf_eur['actions_own_p90']:.1f}"),
        ], ps, col_w),
        _pdf_dark_card(tones[1], "Duels", [
            ("Defensive Duels Per Game", f"{d_def['duels_p90']:.1f}", "league", f"{T_pdf_mls['duels_p90']:.1f}", f"{T_pdf_eur['duels_p90']:.1f}"),
            ("% Duels Won", f"{d_def['duels_won_pct']:.1f}%", "league", f"{T_pdf_mls['duels_won_pct']:.1f}%", f"{T_pdf_eur['duels_won_pct']:.1f}%"),
        ], ps, col_w),
        _pdf_dark_card(tones[2], "Funnel Protection", [
            ("Funnel Protection Actions Per Game", f"{d_def['funnel_actions_p90']:.1f}", "sga", SGA_RANGE_METRICS["funnel_actions_p90"]),
            ("% FPA Successful", f"{d_def['funnel_success_pct']:.1f}%", "league", f"{T_pdf_mls['funnel_success_pct']:.1f}%", f"{T_pdf_eur['funnel_success_pct']:.1f}%"),
        ], ps, col_w),
    ]

    story = [
        Paragraph("Hudson Cicala — Season Dashboard", ps["title"]),
        Paragraph("2026 Season • All Matches Report", ps["sub"]),
        Spacer(1, 0.08 * cm),
    ]
    story.extend(_pdf_dashboard_section("Passing Analysis", pass_maps, pass_cards, ps))
    story.append(PageBreak())
    story.append(Paragraph("Hudson Cicala — Season Dashboard", ps["title"]))
    story.append(Paragraph("2026 Season • All Matches Report", ps["sub"]))
    story.append(Spacer(1, 0.08 * cm))
    story.extend(_pdf_dashboard_section("Defensive Actions", def_maps, def_cards, ps))

    doc.build(story, onFirstPage=_pdf_draw_dark_page, onLaterPages=_pdf_draw_dark_page)
    buf.seek(0)
    return buf.getvalue()

# ── SIDEBAR ────────────────────────────────────────────────────
st.sidebar.markdown("""
<h2 style="color:#ffffff;font-weight:700;margin-bottom:4px">Pass Stats Dashboard</h2>
<p style="color:#aaaaaa;font-size:13px;margin-top:0">2026 Season • Hudson Cicala</p>
""", unsafe_allow_html=True)

img_path = "Captura de tela 2026-06-02 154425.png"
if os.path.exists(img_path):
    st.sidebar.image(img_path, use_container_width=True)

st.sidebar.markdown("""
<p style="color:#666666;font-size:11px;margin-top:8px">Data collected from match footage</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### Stats Card Style")
CARD_STYLE = st.sidebar.radio(
    "Choose visual layout",
    options=list(TARGET_CARD_STYLES.keys()),
    index=0,
    help="Informative scoreboard layouts. Value inline with label; MLS and TOP 5 EUR benchmarks below.",
)
CARD_LAYOUT = st.sidebar.radio(
    "Card separation",
    options=["Combined", "Separated"],
    index=0,
    help="Separated shows player values in one card and benchmark references in a second card below.",
)
_layout_mode = "separated" if CARD_LAYOUT == "Separated" else "combined"

st.sidebar.markdown("#### Card Color")
CARD_TONE_SCHEME = st.sidebar.radio(
    "Accent palette",
    options=["Blue", "Gray"],
    index=0,
    help="Blue uses the default accent tones; Gray uses neutral gray card borders.",
)
ACTIVE_CARD_TONES = GRAY_TONES if CARD_TONE_SCHEME == "Gray" else PASS_TONES

st.sidebar.markdown("#### Target Benchmark")
TARGET_BENCHMARK = st.sidebar.radio(
    "Comparison pool",
    options=["MLS", BENCHMARK_EUR_KEY],
    index=0,
    help=(
        "Targets for passes and defensive duels are the position-filtered averages "
        f"from the selected database ({', '.join(BENCHMARK_POSITIONS)}; "
        f"≥{int(BENCHMARK_MINUTES_RATIO * 100)}% of max minutes)."
    ),
)
_bench_meta = load_benchmark_targets(TARGET_BENCHMARK)
if _bench_meta is None:
    st.sidebar.warning(f"Benchmark file not found for {TARGET_BENCHMARK}. Using fallback targets.")
else:
    st.sidebar.caption(
        f"{TARGET_BENCHMARK}: {_bench_meta['sample_size']} players · "
        f"≥{_bench_meta['minutes_threshold']:.0f} min"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("#### Export Report")
if not PDF_AVAILABLE:
    st.sidebar.warning("Install reportlab to enable PDF export: `pip install reportlab`")
else:
    if st.sidebar.button("Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Building season report..."):
            try:
                pdf_bytes = generate_season_pdf(
                    card_tones=ACTIVE_CARD_TONES,
                    benchmark_source=TARGET_BENCHMARK,
                )
                st.session_state["pdf_report"] = pdf_bytes
                st.sidebar.success("Report ready!")
            except Exception as exc:
                st.sidebar.error(f"PDF error: {exc}")
    if "pdf_report" in st.session_state:
        st.sidebar.download_button(
            label="Download PDF",
            data=st.session_state["pdf_report"],
            file_name="Hudson_Cicala_Season_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.sidebar.caption("All Matches • Passes & Defensive Actions")

num_matches = len(dfs_by_match)
all_match_stats = [compute_stats(dfs_by_match[m], m) for m in dfs_by_match]
def_all_stats = [compute_defensive_stats(defensive_dfs_by_match[m], m) for m in defensive_dfs_by_match]

_pass_base = {}
if num_matches > 0:
    for k in all_match_stats[0].keys():
        if isinstance(all_match_stats[0][k], (int, float)):
            _pass_base[k] = sum(s[k] for s in all_match_stats) / num_matches

_def_base = {}
if len(def_all_stats) > 0:
    for k in def_all_stats[0].keys():
        if isinstance(def_all_stats[0][k], (int, float)):
            _def_base[k] = sum(s[k] for s in def_all_stats) / len(def_all_stats)

T_MLS = build_metric_targets(_pass_base, _def_base, "MLS")
T_TOP_EUR = build_metric_targets(_pass_base, _def_base, BENCHMARK_EUR_KEY)
METRIC_TARGETS = T_MLS if TARGET_BENCHMARK == "MLS" else T_TOP_EUR
T = METRIC_TARGETS

with st.sidebar.expander("Season targets (reference)"):
    _tgt_rows = []
    _db_keys = {
        "total_p90", "accuracy_pct", "advanced_passes_p90", "advanced_accuracy_pct",
        "total_actions_p90", "duels_p90", "duels_won_pct",
    }
    for _k, _v in T_MLS.items():
        _base = _pass_base.get(_k, _def_base.get(_k, 0))
        if _k in SGA_RANGE_METRICS:
            _ref = f"Range SGA: {SGA_RANGE_METRICS[_k]}"
            _mls_ref = _ref
            _eur_ref = _ref
        else:
            _mls_ref = T_MLS[_k]
            _eur_ref = T_TOP_EUR[_k]
        _tgt_rows.append({
            "Metric": _k,
            "Hudson Per Game": round(_base, 2),
            "MLS": _mls_ref,
            "TOP 5 EUR": _eur_ref,
        })
    st.dataframe(pd.DataFrame(_tgt_rows), hide_index=True, use_container_width=True)

# ── LAYOUT ─────────────────────────────────────────────────────
tab_dash, = st.tabs(["Detailed Dashboard"])

with tab_dash:
    sub_tab_passes, sub_tab_def = st.tabs(["Passes", "Defensive Actions"])

    # ═══════════════════════════════════════════════════════════
    # PASSES TAB
    # ═══════════════════════════════════════════════════════════
    with sub_tab_passes:
        st.markdown("### Match Filters")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            pass_match_options = ["All Matches"] + list(dfs_by_match.keys())
            selected_match = st.selectbox("Select Match", options=pass_match_options, index=0, key="pass_match")
        with col_f2:
            pass_filter = st.radio("Pass Type", ["All", "Successful", "Unsuccessful", "Progressive", "Final Third"],
                                   index=0, horizontal=True, key="pass_filter")

        if selected_match == "All Matches":
            df_game_filtered = pd.concat(dfs_by_match.values(), ignore_index=True)
            match_name_for_stats = "All Matches"
        else:
            df_game_filtered = dfs_by_match[selected_match].copy()
            match_name_for_stats = selected_match

        def apply_filter(df):
            if pass_filter == "Successful":
                return df[df["is_won"]].copy()
            if pass_filter == "Unsuccessful":
                return df[~df["is_won"]].copy()
            if pass_filter == "Progressive":
                return df[df["progressive"]].copy()
            if pass_filter == "Final Third":
                return df[(df["x_start"] < FINAL_THIRD_LINE_X) & (df["x_end"] >= FINAL_THIRD_LINE_X)].copy()
            return df.copy()

        df_game = apply_filter(df_game_filtered)
        s_game = compute_stats(df_game, match_name_for_stats)
        s_avg = {}
        if num_matches > 0:
            for k in all_match_stats[0].keys():
                if isinstance(all_match_stats[0][k], (int, float)):
                    s_avg[k] = sum(s[k] for s in all_match_stats) / num_matches
                else:
                    s_avg[k] = 0
        else:
            s_avg = s_game.copy()

        force_avg = selected_match == "All Matches"
        if force_avg:
            s_game = s_avg.copy()

        st.markdown("---")

        img_pm_game, fig_pm_game = draw_pass_map(df_game); plt.close(fig_pm_game)
        img_ht_game, fig_ht_game = draw_corridor_heatmap(df_game); plt.close(fig_ht_game)

        top_n_xt = 10 if force_avg else 5
        img_xt_game, fig_xt_game = draw_top_xt_map(df_game, top_n=top_n_xt)

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown('<p style="font-size:13px;font-weight:600;color:#cccccc">Pass Map</p>', unsafe_allow_html=True)
            st.image(img_pm_game,use_container_width=True)
        with col_m2:
            st.markdown('<p style="font-size:13px;font-weight:600;color:#cccccc">Zone Heatmap (Destination)</p>', unsafe_allow_html=True)
            st.image(img_ht_game,use_container_width=True)
        with col_m3:
            label="Top 10" if force_avg else "Top 5"
            st.markdown(f'<p style="font-size:13px;font-weight:600;color:#cccccc">{label} Pass Impact</p>', unsafe_allow_html=True)
            st.pyplot(fig_xt_game, use_container_width=True)
            plt.close(fig_xt_game)

        st.markdown("<hr style='margin:8px 0;opacity:0.2'>", unsafe_allow_html=True)

        # ── STATS CARDS ───────────────────────────────────────
        col_s1, col_s2, col_s3 = st.columns(3)

        with col_s1:
            target_section_card("Overview", ACTIVE_CARD_TONES[0], [
                build_metric_item("Total Passes Per Game", s_game["total_p90"], f"{s_game['total_p90']:.1f}", "total_p90"),
                build_metric_item("% Accuracy", s_game["accuracy_pct"], f"{s_game['accuracy_pct']:.1f}%", "accuracy_pct"),
            ], CARD_STYLE, _layout_mode, export_key="pass_overview_png")
        with col_s2:
            target_section_card("Progressive", ACTIVE_CARD_TONES[1], [
                build_metric_item("Progressive Passes Per Game", s_game["advanced_passes_p90"], f"{s_game['advanced_passes_p90']:.1f}", "advanced_passes_p90"),
                build_metric_item("% Progressive Accuracy", s_game["advanced_accuracy_pct"], f"{s_game['advanced_accuracy_pct']:.1f}%", "advanced_accuracy_pct"),
            ], CARD_STYLE, _layout_mode, export_key="pass_progressive_png")
        with col_s3:
            target_section_card("Impact", ACTIVE_CARD_TONES[2], [
                build_metric_item("Pass Impact Value Per Game", s_game["xt_p90"], f"{s_game['xt_p90']:.1f}", "xt_p90"),
                build_metric_item("% Positive Impact", s_game["pos_pct"], f"{s_game['pos_pct']:.1f}%", "pos_pct"),
            ], CARD_STYLE, _layout_mode, export_key="pass_impact_png")
    # ═══════════════════════════════════════════════════════════
    with sub_tab_def:
        st.markdown("### Match Filter")
        col_df1, col_df2 = st.columns(2)
        with col_df1:
            def_match_options=["All Matches"]+list(defensive_dfs_by_match.keys())
            selected_def_match=st.selectbox("Select Match",options=def_match_options,index=0,key="def_match")
        with col_df2:
            def_type_filter=st.radio("Filter Type",["All","Duels Only","Interceptions Only"],horizontal=True,key="def_type_filter")

        if selected_def_match=="All Matches":
            df_def_game_raw=pd.concat(defensive_dfs_by_match.values(),ignore_index=True)
            def_match_name_for_stats="All Matches"
        else:
            df_def_game_raw=defensive_dfs_by_match[selected_def_match].copy()
            def_match_name_for_stats=selected_def_match

        if def_type_filter=="Duels Only":
            df_def_game=df_def_game_raw[df_def_game_raw["is_duel"]].copy()
        elif def_type_filter=="Interceptions Only":
            df_def_game=df_def_game_raw[df_def_game_raw["is_interception"]].copy()
        else:
            df_def_game=df_def_game_raw.copy()

        d_game=compute_defensive_stats(df_def_game,def_match_name_for_stats)
        def_all=[compute_defensive_stats(defensive_dfs_by_match[m],m) for m in defensive_dfs_by_match]
        d_avg={}
        if len(def_all)>0:
            for k in def_all[0].keys():
                if isinstance(def_all[0][k],(int,float)):
                    d_avg[k]=sum(s[k] for s in def_all)/len(def_all)
                else:
                    d_avg[k]=0
        else:
            d_avg=d_game.copy()

        force_avg_def=selected_def_match=="All Matches"
        if force_avg_def:
            d_game=d_avg.copy()

        st.markdown("---")

        img_def_map,fig_def_map=draw_defensive_map(df_def_game); plt.close(fig_def_map)
        img_def_hm,fig_def_hm=draw_defensive_heatmap(df_def_game); plt.close(fig_def_hm)
        img_funnel,fig_funnel=draw_funnel_protection_map(df_def_game)

        col_dm1,col_dm2,col_dm3=st.columns(3)
        with col_dm1:
            st.markdown('<p style="font-size:13px;font-weight:600;color:#cccccc">Defensive Actions Map</p>', unsafe_allow_html=True)
            st.image(img_def_map,use_container_width=True)
        with col_dm2:
            st.markdown('<p style="font-size:13px;font-weight:600;color:#cccccc">Defensive Heatmap</p>', unsafe_allow_html=True)
            st.image(img_def_hm,use_container_width=True)
        with col_dm3:
            st.markdown('<p style="font-size:13px;font-weight:600;color:#cccccc">Funnel Protection Actions</p>', unsafe_allow_html=True)
            st.pyplot(fig_funnel, use_container_width=True)
            plt.close(fig_funnel)

        st.markdown("<hr style='margin:8px 0;opacity:0.2'>", unsafe_allow_html=True)

        # ── DEFENSIVE STATS CARDS ─────────────────────────────
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            target_section_card("Overview", ACTIVE_CARD_TONES[0], [
                build_metric_item("Defensive Actions Per Game", d_game["total_actions_p90"], f"{d_game['total_actions_p90']:.1f}", "total_actions_p90"),
                build_metric_item("Actions in Own Half Per Game", d_game["actions_own_p90"], f"{d_game['actions_own_p90']:.1f}", "actions_own_p90"),
            ], CARD_STYLE, _layout_mode, export_key="def_overview_png")
        with col_ds2:
            target_section_card("Duels", ACTIVE_CARD_TONES[1], [
                build_metric_item("Defensive Duels Per Game", d_game["duels_p90"], f"{d_game['duels_p90']:.1f}", "duels_p90"),
                build_metric_item("% Duels Won", d_game["duels_won_pct"], f"{d_game['duels_won_pct']:.1f}%", "duels_won_pct"),
            ], CARD_STYLE, _layout_mode, export_key="def_duels_png")
        with col_ds3:
            target_section_card("Funnel Protection", ACTIVE_CARD_TONES[2], [
                build_metric_item("Funnel Protection Actions Per Game", d_game["funnel_actions_p90"], f"{d_game['funnel_actions_p90']:.1f}", "funnel_actions_p90"),
                build_metric_item("% FPA Successful", d_game["funnel_success_pct"], f"{d_game['funnel_success_pct']:.1f}%", "funnel_success_pct"),
            ], CARD_STYLE, _layout_mode, export_key="def_funnel_png")
