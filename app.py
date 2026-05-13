import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(
    page_title="NeuroScan AI · Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #050A0F !important;
    color: #C8D8E8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { background: #050A0F !important; }
[data-testid="stSidebar"] {
    background: #070D14 !important;
    border-right: 1px solid #0D2035 !important;
}
[data-testid="stSidebar"] * { color: #8BA4BC !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #050A0F; }
::-webkit-scrollbar-thumb { background: #0A4A7A; border-radius: 2px; }

.stButton > button {
    background: linear-gradient(135deg, #0A3A5A, #0A5A8A) !important;
    color: #00C2FF !important;
    border: 1px solid #0A6AAA !important;
    border-radius: 10px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 14px 36px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    box-shadow: 0 0 20px rgba(10,90,138,0.4) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0A5A8A, #0A7ABE) !important;
    box-shadow: 0 0 32px rgba(0,194,255,0.35) !important;
    transform: translateY(-1px) !important;
}
.stProgress > div > div > div {
    background: #0A1A2A !important;
    border-radius: 3px !important;
    height: 6px !important;
}
.stProgress > div > div > div > div {
    border-radius: 3px !important;
}
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #0D2035 !important;
}
[data-testid="column"] { padding: 0 10px !important; }
</style>
""", unsafe_allow_html=True)

# ── Model — architecture EXACTE du checkpoint ────────────────────────────────
# backbone.classifier:
#   0: Dropout(0.2)
#   1: Linear(1280, 512)
#   2: ReLU
#   3: BatchNorm1d(512)
#   4: Dropout(0.2)   ← clé manquante dans les versions précédentes !
#   5: Linear(512, 4)
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.efficientnet_b0(pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),  # 0
            nn.Linear(in_features, 512),       # 1
            nn.ReLU(),                         # 2
            nn.BatchNorm1d(512),               # 3
            nn.Dropout(p=0.2),                 # 4  ← correction clé
            nn.Linear(512, num_classes)        # 5
        )
    def forward(self, x): return self.backbone(x)

@st.cache_resource
def load_all_models():
    yolo = YOLO('yolo_best.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn = EfficientNetClassifier(num_classes=4).to(device)
    cnn.load_state_dict(torch.load('cnn_best.pth', map_location=device), strict=True)
    cnn.eval()
    return yolo, cnn, device

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
COLORS_HEX = {'Glioma':'#FF4444','Meningioma':'#FF8C00','No Tumor':'#00C2FF','Pituitary':'#A855F7'}

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:3px;color:#1A4A6A;text-transform:uppercase;padding:16px 0 12px 0;border-bottom:1px solid #0A1A2A;">⚙ Configuration</p>', unsafe_allow_html=True)
    conf_threshold = st.slider("Detection Threshold", 0.10, 1.00, 0.30, 0.05)
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:3px;color:#1A4A6A;text-transform:uppercase;padding:20px 0 12px 0;border-bottom:1px solid #0A1A2A;">Model Info</p>', unsafe_allow_html=True)
    for label, value in [("DETECTOR","YOLOv8"),("CLASSIFIER","EfficientNet-B0"),("CLASSES","4")]:
        st.markdown(f"""<div style="background:#060D14;border:1px solid #0A1E2E;border-radius:8px;padding:12px 14px;margin-bottom:8px;">
            <div style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#1A4A6A;margin-bottom:3px;">{label}</div>
            <div style="font-family:Space Mono,monospace;font-size:14px;font-weight:700;color:{'#00C2FF' if label=='CLASSES' else '#8BA4BC'};">{value}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#1A4A6A;text-transform:uppercase;margin:20px 0 10px 0;padding-top:16px;border-top:1px solid #0A1A2A;">Classes</p>', unsafe_allow_html=True)
    for cls, col in COLORS_HEX.items():
        st.markdown(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px;"><div style="width:8px;height:8px;border-radius:50%;background:{col};flex-shrink:0;"></div><span style="font-size:12px;color:#5A7A8A;">{cls}</span></div>', unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;padding:24px 0 20px 0;border-bottom:1px solid #0D2035;margin-bottom:28px;">
    <div style="width:48px;height:48px;background:linear-gradient(135deg,#0A4A7A,#00C2FF);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;box-shadow:0 0 24px rgba(0,194,255,0.25);">🧠</div>
    <div>
        <div style="font-family:Space Mono,monospace;font-size:22px;font-weight:700;color:#FFFFFF;">NeuroScan AI</div>
        <div style="font-size:11px;color:#3A6A8A;letter-spacing:2px;text-transform:uppercase;">Brain Tumor Detection &amp; Classification System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<p style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:3px;color:#1A4A6A;text-transform:uppercase;margin-bottom:10px;padding-bottom:8px;border-bottom:1px solid #0A1A2A;">01 · Upload MRI Scan</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop an MRI image here", type=["jpg","png","jpeg"], label_visibility="collapsed")

if not uploaded_file:
    st.markdown('<div style="text-align:center;padding:40px 0;"><div style="font-size:40px;margin-bottom:12px;opacity:0.2;">⬆</div><div style="font-family:Space Mono,monospace;font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#1A3A5A;">Upload a JPEG or PNG MRI scan to begin</div></div>', unsafe_allow_html=True)
    st.stop()

yolo_model, cnn_model, DEVICE = load_all_models()
img_pil = Image.open(uploaded_file).convert('RGB')

# ── Scan preview ──────────────────────────────────────────────────────────────
st.markdown('<p style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:3px;color:#1A4A6A;text-transform:uppercase;margin:24px 0 10px 0;padding-bottom:8px;border-bottom:1px solid #0A1A2A;">02 · Scan Preview</p>', unsafe_allow_html=True)
col_orig, col_result = st.columns(2, gap="medium")
with col_orig:
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#1A4A6A;text-transform:uppercase;margin-bottom:6px;">Original Scan</p>', unsafe_allow_html=True)
    st.image(img_pil, use_container_width=True)
result_slot = col_result.empty()
with result_slot.container():
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#1A4A6A;text-transform:uppercase;margin-bottom:6px;">Detection Result</p>', unsafe_allow_html=True)
    st.image(img_pil, use_container_width=True)

st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
btn_col, _ = st.columns([1, 2])
with btn_col:
    run = st.button("▶  Run Analysis Pipeline", use_container_width=True)
if not run:
    st.stop()

# ── Pipeline ──────────────────────────────────────────────────────────────────
def classify_region(region_img):
    t = val_transforms(region_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(t), dim=1)[0]
        idx = torch.argmax(probs).item()
    return CLASSES[idx], probs[idx].item(), probs

with st.spinner("Running YOLO detection…"):
    boxes = yolo_model(img_pil, conf=conf_threshold)[0].boxes

img_draw = img_pil.copy()
draw = ImageDraw.Draw(img_draw)
best_label, best_conf, best_probs, best_box = None, 0.0, None, None
used_fallback = False

# CNN classifies the full image for the most reliable diagnosis
best_label, best_conf, best_probs = classify_region(img_pil)
c = COLORS_HEX[best_label]

if len(boxes) > 0:
    # Collect and sort YOLO boxes by confidence
    yolo_boxes = sorted(
        [tuple(map(int, box.xyxy[0].tolist())) + (float(box.conf[0]),) for box in boxes],
        key=lambda b: b[4], reverse=True
    )
    # Draw all boxes — strongest one gets the label
    for i, (x1,y1,x2,y2,yc) in enumerate(yolo_boxes):
        draw.rectangle([x1,y1,x2,y2], outline=c if i==0 else "#2A5A7A", width=4 if i==0 else 2)
    # Label on best box
    x1,y1,x2,y2,_ = yolo_boxes[0]
    tag = f" {best_label} {best_conf:.0%} "
    tw = len(tag) * 8
    lx, ly = x1, max(y1-22, 0)
    draw.rectangle([lx, ly, lx+tw, ly+20], fill=c)
    draw.text((lx+4, ly+3), tag.strip(), fill="#000000")
    used_fallback = False
else:
    # No YOLO boxes — frame the whole image
    used_fallback = True
    w, h = img_pil.size
    draw.rectangle([5,5,w-5,h-5], outline=c, width=3)
    tag = f" {best_label} {best_conf:.0%} "
    tw = len(tag) * 8
    draw.rectangle([5,5,5+tw,27], fill=c)
    draw.text((10,8), tag.strip(), fill="#000000")

with result_slot.container():
    st.markdown('<p style="font-size:10px;letter-spacing:2px;color:#1A4A6A;text-transform:uppercase;margin-bottom:6px;">Detection Result</p>', unsafe_allow_html=True)
    st.image(img_draw, use_container_width=True)

# ── Report ────────────────────────────────────────────────────────────────────
st.markdown('<p style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:3px;color:#1A4A6A;text-transform:uppercase;margin:32px 0 16px 0;padding-bottom:8px;border-bottom:1px solid #0A1A2A;">03 · Analysis Report</p>', unsafe_allow_html=True)

is_tumor = best_label != 'No Tumor'
diag_color = COLORS_HEX[best_label]
top_bar = "linear-gradient(90deg,#FF4444,#FF8800)" if is_tumor else "linear-gradient(90deg,#00C2FF,#00FF88)"
status_color = "#FF6B6B" if is_tumor else "#00C2FF"

diag_col, scores_col = st.columns(2, gap="medium")

with diag_col:
    st.markdown(f"""
    <div style="background:linear-gradient(145deg,#080F18,#060C14);border-radius:16px;padding:28px;border:1px solid #0D2035;position:relative;overflow:hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{top_bar};border-radius:16px 16px 0 0;"></div>
        <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;color:#3A6A8A;font-family:Space Mono,monospace;margin-bottom:8px;">Primary Diagnosis</div>
        <div style="font-family:Space Mono,monospace;font-size:36px;font-weight:700;color:{diag_color};line-height:1.1;margin-bottom:6px;">{best_label}</div>
        <div style="font-size:13px;color:#3A6A8A;margin-bottom:24px;">Confidence: {best_conf:.1%}</div>
        <div style="border-top:1px solid #0A1A2A;padding-top:16px;display:flex;gap:32px;">
            <div>
                <div style="font-size:10px;letter-spacing:2px;color:#1A4A6A;text-transform:uppercase;margin-bottom:4px;">Status</div>
                <div style="font-family:Space Mono,monospace;font-size:12px;color:{status_color};">{'TUMOR DETECTED' if is_tumor else 'CLEAR'}</div>
            </div>
            <div>
                <div style="font-size:10px;letter-spacing:2px;color:#1A4A6A;text-transform:uppercase;margin-bottom:4px;">Method</div>
                <div style="font-family:Space Mono,monospace;font-size:12px;color:#3A6A8A;">{'YOLO + CNN' if not used_fallback else 'CNN FALLBACK'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if used_fallback:
        st.markdown('<div style="background:#060E18;border:1px solid #0A3050;border-left:3px solid #0A6AAA;border-radius:8px;padding:10px 14px;margin-top:12px;font-size:11px;color:#5A8AAA;font-family:Space Mono,monospace;">ⓘ YOLO found no reliable region — full image classified by CNN.</div>', unsafe_allow_html=True)

with scores_col:
    st.markdown('<div style="background:#080F18;border:1px solid #0D2035;border-radius:16px;padding:24px;">', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:10px;letter-spacing:3px;color:#1A4A6A;text-transform:uppercase;margin-bottom:16px;">Confidence Scores</p>', unsafe_allow_html=True)
    for i, cls in enumerate(CLASSES):
        p = float(best_probs[i])
        dot = COLORS_HEX[cls]
        is_best = cls == best_label
        lbl_style = f"font-weight:600;color:{dot};" if is_best else "color:#5A8AAA;"
        pct_style = f"color:{dot};" if is_best else "color:#2A5A7A;"
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:7px;height:7px;border-radius:50%;background:{dot};"></div>
                <span style="font-family:Space Mono,monospace;font-size:11px;{lbl_style}">{cls}</span>
            </div>
            <span style="font-family:Space Mono,monospace;font-size:11px;{pct_style}">{p*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(p)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="margin-top:40px;padding-top:16px;border-top:1px solid #0A1A2A;display:flex;justify-content:space-between;"><span style="font-family:Space Mono,monospace;font-size:10px;color:#0A2A3A;letter-spacing:2px;">NEUROSCAN AI · FOR RESEARCH USE ONLY</span><span style="font-family:Space Mono,monospace;font-size:10px;color:#0A2A3A;letter-spacing:1px;">EfficientNet-B0 + YOLOv8</span></div>', unsafe_allow_html=True)