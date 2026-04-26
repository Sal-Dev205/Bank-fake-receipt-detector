"""
Askari Bank — Receipt Fraud Detector
Powered by: Fine-tuned EfficientNet B0 + Tesseract OCR + Rule Engine
No API key required. 100% local.

FIRST-TIME SETUP:
  1. Install Tesseract:
       Windows : https://github.com/UB-Mannheim/tesseract/wiki
       macOS   : brew install tesseract
       Ubuntu  : sudo apt install tesseract-ocr

  2. Install Python deps:
       pip install -r requirements_free.txt

  3. Train the model (run once):
       python train_model.py --data path/to/data
       (data folder must contain 'real' and 'Fake' subfolders)

  4. Launch the app:
       streamlit run askari_receipt_detector_free.py
"""

import os
import re
import io
import math

import streamlit as st
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ── Windows: explicit Tesseract path ──
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MODEL_PATH = "askari_efficientnet.pth"

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Askari Bank — Receipt Fraud Detector",
    page_icon="🏦",
    layout="centered",
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.logo-pill { display:inline-block; background:#0b1f3a; color:white; padding:6px 14px; border-radius:8px; font-size:15px; font-weight:600; }
.logo-pill span { color:#5da8e0; }
.header-sub { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#8a95a3; margin-top:4px; letter-spacing:1.5px; }
.verdict-authentic { background:#f0faf0; border:1px solid #b8ddb8; border-radius:14px; padding:16px 20px; }
.verdict-fake      { background:#fef2f2; border:1px solid #fac5c5; border-radius:14px; padding:16px 20px; }
.verdict-suspicious{ background:#fffbf0; border:1px solid #fde6a0; border-radius:14px; padding:16px 20px; }
.chip-pass { background:#f0faf0; border:1px solid #b8ddb8; color:#2d6e2d; border-radius:8px; padding:6px 10px; font-size:12px; font-weight:500; display:inline-block; margin:3px; }
.chip-fail { background:#fef5f5; border:1px solid #f5c0c0; color:#a32020; border-radius:8px; padding:6px 10px; font-size:12px; font-weight:500; display:inline-block; margin:3px; }
.chip-warn { background:#fffbf0; border:1px solid #f5dea0; color:#8a6200; border-radius:8px; padding:6px 10px; font-size:12px; font-weight:500; display:inline-block; margin:3px; }
.ocr-box { background:#f8f9fa; border:1px solid #e2e6ea; border-radius:8px; padding:12px; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#555; white-space:pre-wrap; max-height:220px; overflow-y:auto; }
.sec-title { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#a0a8b0; letter-spacing:1.5px; text-transform:uppercase; margin-bottom:10px; margin-top:4px; }
.score-card { background:#f8f9fa; border:1px solid #e2e6ea; border-radius:12px; padding:14px 16px; margin-bottom:8px; }
.score-card-title { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#8a95a3; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px; }
.model-badge-ok  { background:#f0faf0; border:1px solid #b8ddb8; color:#2d6e2d; border-radius:6px; padding:4px 10px; font-size:11px; font-family:'IBM Plex Mono',monospace; }
.model-badge-off { background:#fff8f0; border:1px solid #fde6a0; color:#8a6200; border-radius:6px; padding:4px 10px; font-size:11px; font-family:'IBM Plex Mono',monospace; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  EFFICIENTNET MODEL
# ─────────────────────────────────────────────

def _build_model_arch():
    """Build the EfficientNet B0 architecture with our 2-class head."""
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2),
    )
    return model


@st.cache_resource(show_spinner="Loading EfficientNet model…")
def load_model():
    """
    Load the fine-tuned model if it exists, otherwise fall back to
    the pretrained ImageNet weights (proxy mode).
    Returns (model, is_finetuned, class_to_idx, info_str).
    """
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint   = torch.load(MODEL_PATH, map_location="cpu")
            model        = _build_model_arch()
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            c2i          = checkpoint.get("class_to_idx", {"Fake": 0, "real": 1})
            acc          = checkpoint.get("final_accuracy", 0.0)
            epochs       = checkpoint.get("epochs", "?")
            real_idx     = c2i.get("real", 1)
            info         = f"Fine-tuned · {epochs} epochs · train acc {acc:.1f}%"
            return model, True, real_idx, info
        except Exception as e:
            st.warning(f"Could not load fine-tuned model ({e}). Using pretrained weights.")

    # Fallback: pretrained EfficientNet B0 (proxy mode)
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = efficientnet_b0(weights=weights)
    model.eval()
    return model, False, None, "Pretrained ImageNet (proxy mode — run train_model.py to fine-tune)"


EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def visual_score(image: Image.Image, model, is_finetuned: bool, real_idx) -> dict:
    """
    Fine-tuned model   → probability of class 'real' = visual score (0–100).
    Pretrained model   → softmax entropy proxy (same as before).
    """
    img_tensor = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0)

    if is_finetuned:
        with torch.no_grad():
            logits = model(img_tensor)
            probs  = F.softmax(logits, dim=1).squeeze()
        real_prob    = probs[real_idx].item()
        score        = int(real_prob * 100)
        fake_prob    = 1.0 - real_prob
        detail       = (f"P(authentic)={real_prob*100:.1f}%  "
                        f"P(fake)={fake_prob*100:.1f}%")
        return {"visual_score": score, "detail": detail, "mode": "fine-tuned"}

    else:
        # Proxy: entropy of 1000-class softmax + feature std
        features_out = {}
        def hook(m, i, o):
            features_out["feat"] = o.squeeze()
        handle = model.avgpool.register_forward_hook(hook)
        with torch.no_grad():
            logits = model(img_tensor)
        handle.remove()

        probs        = F.softmax(logits, dim=1).squeeze()
        entropy      = -(probs * torch.log(probs + 1e-10)).sum().item()
        norm_entropy = entropy / math.log(1000)
        e_score      = (1.0 - norm_entropy) * 100
        f_score      = min(features_out["feat"].std().item() / 1.0, 1.0) * 100
        score        = int(0.6 * e_score + 0.4 * f_score)
        score        = max(0, min(100, score))
        detail       = (f"Entropy={norm_entropy:.3f}  "
                        f"entropy_score={int(e_score)}%  feat_score={int(f_score)}%")
        return {"visual_score": score, "detail": detail, "mode": "proxy"}


# ─────────────────────────────────────────────
#  RULE ENGINE
#  Rules derived from analysis of real Askari receipts.
#  Pattern rules:
#    • Heading must be "FUNDS TRANSFERRED"
#    • Bank name must be "askaribank"
#    • Transaction ID label: "Transaction reference ID #" must appear
#    • Transaction ID value: exactly 12 consecutive digits, no letters/symbols
#    • Date: DD-Mon-YYYY HH:MM:SS AM/PM  (12-hour, named month)
#    • Transfer type: ONLY "Within Askari" or "Interbank" (no other variants)
#    • Within Askari → beneficiary bank must be Askari Bank; no external TO BANK
#    • Interbank → TO BANK field must contain a valid SBP-licensed bank
#    • Amount: numeric, no PKR / Rs symbol
# ─────────────────────────────────────────────

VALID_BANKS = [
    'allied bank', 'hbl', 'habib bank', 'mcb', 'united bank', 'ubl',
    'meezan bank', 'bank al-falah', 'bank alfalah', 'bank al falah',
    'bank al-habib', 'bank alhabib', 'faysal bank', 'standard chartered',
    'js bank', 'nbp', 'national bank', 'soneri bank', 'silk bank',
    'habib metro', 'askari bank', 'summit bank', 'bankislami', 'bank islami',
    'al baraka', 'dubai islamic', 'first women bank', 'samba bank',
    'sindh bank', 'the bank of punjab', 'bop', 'zarai taraqiati',
    'ztbl', 'industrial development bank', 'idbp', 'khushhali bank',
]
SBP_RE = re.compile('|'.join(VALID_BANKS), re.IGNORECASE)

MONTHS = {
    'jan': 31, 'feb': 28, 'mar': 31, 'apr': 30,
    'may': 31, 'jun': 30, 'jul': 31, 'aug': 31,
    'sep': 30, 'oct': 31, 'nov': 30, 'dec': 31,
}


def _leap(y):
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def _validate_date(s):
    m = re.search(
        r'(\d{1,2})-([A-Za-z]{3})-(\d{4})\s+(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)',
        s, re.IGNORECASE
    )
    if not m:
        return False, 'Format wrong — expected DD-Mon-YYYY HH:MM:SS AM/PM'
    d, mon, y = int(m.group(1)), m.group(2).lower(), int(m.group(3))
    hr, mn, sc = int(m.group(4)), int(m.group(5)), int(m.group(6))
    if mon not in MONTHS:
        return False, f'Unknown month abbreviation: {m.group(2)}'
    max_d = 29 if (mon == 'feb' and _leap(y)) else MONTHS[mon]
    if not (1 <= d <= max_d):
        return False, f'Day {d} invalid for {m.group(2)}'
    if not (1 <= hr <= 12):
        return False, f'Hour {hr} invalid for 12-hour clock'
    if mn > 59 or sc > 59:
        return False, 'Invalid minutes or seconds'
    if not (2000 <= y <= 2099):
        return False, f'Year {y} out of expected range'
    return True, m.group(0)


# ── Individual checks ──

def check_heading(text):
    if re.search(r'FUNDS\s+TRANSFERRED', text, re.IGNORECASE):
        return 'pass', '"FUNDS TRANSFERRED" heading found'
    return 'fail', '"FUNDS TRANSFERRED" heading missing — wrong bank or heavily edited'


def check_logo(text):
    if re.search(r'askari\s*bank', text, re.IGNORECASE):
        return 'pass', 'Askari Bank name detected'
    if re.search(r'askari', text, re.IGNORECASE):
        return 'warn', '"Askari" found but "Bank" not clearly readable'
    return 'fail', 'Askari Bank name not detected'


def check_transaction_id_label(text):
    """
    The label 'Transaction reference ID #' must appear explicitly.
    Fakes often just put '#032018576677' with no label.
    """
    if re.search(r'Transaction\s+reference\s+ID\s+#', text, re.IGNORECASE):
        return 'pass', 'Transaction reference ID label present'
    if re.search(r'^\s*#\d{10,}', text, re.MULTILINE):
        return 'fail', ('Transaction ID found but missing required label '
                        '"Transaction reference ID #" — common in fake receipts')
    return 'warn', 'Transaction reference ID label not clearly readable'


def check_transaction_id_digits(text):
    """
    The ID after the label must be exactly 12 consecutive digits — no letters,
    no @ symbols, no spaces inside.  (e.g. 11@213A3B4018 → FAIL)
    """
    m = re.search(r'Transaction\s+reference\s+ID\s+#\s*([^\n]{6,20})', text, re.IGNORECASE)
    if not m:
        return 'warn', 'Could not locate transaction ID value to verify'
    raw_id = m.group(1).strip()
    # Remove spaces (some OCR splits digits)
    digits_only = re.sub(r'\s', '', raw_id)
    if not re.fullmatch(r'\d{12}', digits_only):
        non_digit = re.sub(r'\d', '', digits_only)
        return 'fail', (f'Transaction ID "{raw_id}" is not exactly 12 digits. '
                        f'Non-digit characters found: "{non_digit}"')
    return 'pass', f'Transaction ID "{digits_only}" — 12 digits, all numeric ✓'


def check_date_format(text):
    if re.search(r'\d{2}/\d{2}/\d{4}', text):
        return 'fail', 'Slash-format date (DD/MM/YYYY) detected — Askari uses DD-Mon-YYYY only'
    if re.search(r'\b([01]\d|2[0-3]):[0-5]\d:[0-5]\d\b(?!\s*[AaPp][Mm])', text):
        return 'fail', '24-hour clock detected — Askari always uses 12-hour AM/PM format'
    m = re.search(
        r'\d{1,2}-[A-Za-z]{3}-\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*[AaPp][Mm]', text
    )
    if not m:
        return 'warn', 'Date not clearly readable (OCR may have missed it)'
    ok, msg = _validate_date(m.group(0))
    if not ok:
        return 'fail', msg
    return 'pass', f'Date format correct: {m.group(0)}'


def check_transfer_type(text):
    """
    ONLY 'Within Askari' and 'Interbank' are valid.
    'Askari to Askari', 'Intra bank', 'Internal', etc. are FAKE.
    """
    m = re.search(r'TRANSFER\s+TYPE\s+(.+)', text, re.IGNORECASE)
    if not m:
        return 'warn', 'Transfer type field not found'
    raw = m.group(1).strip().split('\n')[0].strip()

    if re.search(r'^within\s+askari$', raw, re.IGNORECASE):
        return 'pass', 'Transfer type: Within Askari ✓'
    if re.search(r'^interbank$', raw, re.IGNORECASE):
        return 'pass', 'Transfer type: Interbank ✓'

    # Any other value is a FAIL
    return 'fail', (f'Transfer type "{raw}" is invalid. '
                    f'Only "Within Askari" or "Interbank" are accepted — '
                    f'variants like "Askari to Askari" indicate a fake.')


def check_bank_consistency(text):
    """
    Within Askari  → no external TO BANK; beneficiary shows Askari Bank.
    Interbank      → TO BANK must be a recognised SBP-licensed bank.
    """
    is_within = bool(re.search(r'within\s+askari', text, re.IGNORECASE))
    is_inter  = bool(re.search(r'\binterbank\b', text, re.IGNORECASE))

    if is_within:
        to_bank_m = re.search(r'TO\s+BANK\s+(.+)', text, re.IGNORECASE)
        if to_bank_m:
            tb = to_bank_m.group(1).strip().split('\n')[0]
            if not re.search(r'askari', tb, re.IGNORECASE):
                return 'fail', f'Within Askari transfer but TO BANK shows "{tb}" (external bank)'
        benef = re.search(r'BENEFICIARY.*\n.*\n(.+)', text, re.IGNORECASE)
        if benef and not re.search(r'askari', benef.group(1), re.IGNORECASE):
            return 'warn', 'Beneficiary bank name may not be Askari Bank'
        return 'pass', 'Within Askari: beneficiary is Askari Bank ✓'

    if is_inter:
        to_bank_m = re.search(r'TO\s+BANK\s+(.+)', text, re.IGNORECASE)
        if not to_bank_m:
            return 'fail', 'Interbank transfer but TO BANK field is missing'
        tb = to_bank_m.group(1).strip().split('\n')[0]
        if not SBP_RE.search(tb):
            return 'warn', f'TO BANK "{tb}" not in list of known SBP-licensed banks'
        return 'pass', f'Interbank TO BANK: {tb} ✓'

    return 'warn', 'Transfer type not readable — cannot verify bank consistency'


def check_currency(text):
    if re.search(r'\bPKR\b|\bRs\.?\s*\d|\bRs\b', text, re.IGNORECASE):
        return 'fail', 'PKR/Rs currency symbol present — not used in authentic Askari receipts'
    return 'pass', 'No incorrect currency symbol detected ✓'


def check_amount(text):
    m = re.search(r'AMOUNT\s*\n([^\n]+)', text, re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        if re.search(r'PKR|Rs\.?', val, re.IGNORECASE):
            return 'fail', f'Amount field "{val}" contains PKR/Rs symbol'
        if re.search(r'\d', val):
            return 'pass', f'Amount field: {val} ✓'
    if re.search(r'\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b', text):
        return 'pass', 'Numeric amount found in receipt ✓'
    return 'warn', 'Amount not clearly readable'


# ── Run all checks ──

CHECKS = [
    ('Bank heading',              check_heading),
    ('Askari Bank logo',          check_logo),
    ('Transaction ID label',      check_transaction_id_label),
    ('Transaction ID digits',     check_transaction_id_digits),
    ('Date format',               check_date_format),
    ('Transfer type',             check_transfer_type),
    ('Bank consistency',          check_bank_consistency),
    ('Currency symbol',           check_currency),
    ('Amount field',              check_amount),
]


def run_ocr_checks(text: str) -> dict:
    results     = []
    fail_count  = warn_count = pass_count = 0
    for label, fn in CHECKS:
        try:
            status, detail = fn(text)
        except Exception as e:
            status, detail = 'warn', f'Check error: {e}'
        if status == 'fail':
            fail_count += 1
        elif status == 'warn':
            warn_count += 1
        else:
            pass_count += 1
        results.append({'label': label, 'status': status, 'detail': detail})

    ocr_score = max(0, min(100, 100 - fail_count * 15 - warn_count * 5))
    return {
        'ocr_score' : ocr_score,
        'checks'    : results,
        'fail_count': fail_count,
        'warn_count': warn_count,
        'pass_count': pass_count,
    }


# ─────────────────────────────────────────────
#  COMBINED VERDICT
# ─────────────────────────────────────────────

def combined_verdict(ocr: dict, vis_score: int, is_finetuned: bool) -> dict:
    """
    Fine-tuned model: 55% OCR + 45% visual (model is specifically trained).
    Proxy model:      70% OCR + 30% visual (OCR rules are more reliable).
    """
    ocr_score = ocr['ocr_score']
    if is_finetuned:
        final = int(0.55 * ocr_score + 0.45 * vis_score)
    else:
        final = int(0.70 * ocr_score + 0.30 * vis_score)
    final = max(0, min(100, final))

    fc, wc, pc = ocr['fail_count'], ocr['warn_count'], ocr['pass_count']

    if fc >= 2 or final < 40:
        verdict, title = 'fake', 'Forgery detected'
        summary = (f'{fc} critical rule check{"s" if fc != 1 else ""} failed. '
                   f'Combined score {final}%. Strong fraud indicators present.')
    elif fc == 1 or wc >= 3 or final < 62:
        verdict, title = 'suspicious', 'Suspicious — manual review needed'
        summary = f'One or more fields inconsistent with Askari standards. Combined score {final}%.'
    elif wc >= 1 or final < 85:
        verdict, title = 'suspicious', 'Mostly authentic — minor issues'
        summary = f'{wc} item{"s" if wc != 1 else ""} unverifiable. Combined score {final}%.'
    else:
        verdict, title = 'authentic', 'Receipt appears authentic'
        summary = f'All {pc} checks passed. Combined score {final}%. No fraud indicators.'

    return {
        'verdict'        : verdict,
        'verdict_title'  : title,
        'verdict_summary': summary,
        'final_score'    : final,
        'ocr_score'      : ocr_score,
        'visual_score'   : vis_score,
        'checks'         : ocr['checks'],
    }


# ─────────────────────────────────────────────
#  OCR PIPELINE
# ─────────────────────────────────────────────

def run_ocr(image: Image.Image) -> str:
    img = image.convert('L')
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(3))
    return pytesseract.image_to_string(img, config=r'--oem 3 --psm 6')


# ─────────────────────────────────────────────
#  RENDER HELPERS
# ─────────────────────────────────────────────

EMOJIS  = {'authentic': '✅', 'suspicious': '⚠️', 'fake': '❌'}
LABELS  = {'authentic': 'VERIFIED', 'suspicious': 'SUSPICIOUS', 'fake': 'FRAUDULENT'}
TCOLORS = {'authentic': '#2d8a2d', 'suspicious': '#b07d0a', 'fake': '#c0392b'}
SEMOJI  = {'pass': '✅', 'warn': '⚠️', 'fail': '❌'}
BCOLOR  = {'authentic': '#4caf50', 'suspicious': '#ffa726', 'fake': '#ef5350'}


def _bar(score, color, label):
    return (
        f'<div class="score-card">'
        f'<div class="score-card-title">{label}</div>'
        f'<div style="display:flex;align-items:center;gap:10px">'
        f'<div style="flex:1;background:#eef0f3;border-radius:4px;height:10px;overflow:hidden;">'
        f'<div style="width:{score}%;height:100%;background:{color};border-radius:4px;"></div>'
        f'</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:14px;font-weight:600;min-width:40px;text-align:right">{score}%</div>'
        f'</div></div>'
    )


def render_verdict(r):
    v = r['verdict']
    st.markdown(
        f'<div class="verdict-{v}">'
        f'<div style="font-size:34px;margin-bottom:6px">{EMOJIS[v]}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;letter-spacing:1.5px;font-weight:600;color:{TCOLORS[v]}">{LABELS[v]}</div>'
        f'<div style="font-size:17px;font-weight:600;color:#1a2533;margin:3px 0">{r["verdict_title"]}</div>'
        f'<div style="font-size:12px;color:#8a95a3;line-height:1.5">{r["verdict_summary"]}</div>'
        f'</div>', unsafe_allow_html=True)


def render_scores(r, is_finetuned, vis_detail, model_info):
    color   = BCOLOR[r['verdict']]
    w_ocr   = "55%" if is_finetuned else "70%"
    w_vis   = "45%" if is_finetuned else "30%"
    mode    = "Fine-tuned" if is_finetuned else "Proxy"
    st.markdown('<div class="sec-title">Scoring Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        _bar(r['final_score'],  color,     f"🔵 Combined Score  ({w_ocr} OCR + {w_vis} EfficientNet)") +
        _bar(r['ocr_score'],    '#1a6fc4', "📋 OCR Rule Score  (9 checks)") +
        _bar(r['visual_score'], '#7b52ab', f"🧠 EfficientNet Visual Score  [{mode}]"),
        unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:11px;color:#a0a8b0;margin-top:4px;font-family:\'IBM Plex Mono\',monospace">'
        f'Visual detail: {vis_detail}<br>Model: {model_info}</div>',
        unsafe_allow_html=True)


def render_checks(checks):
    st.markdown('<div class="sec-title" style="margin-top:16px">Rule Check Results</div>', unsafe_allow_html=True)
    chips = ''.join(
        f'<span class="chip-{c["status"]}">'
        f'{"🟢" if c["status"]=="pass" else "🔴" if c["status"]=="fail" else "🟡"} {c["label"]}'
        f'</span>'
        for c in checks)
    st.markdown(f'<div style="line-height:2.2">{chips}</div>', unsafe_allow_html=True)


def render_findings(checks):
    st.markdown('<div class="sec-title" style="margin-top:16px">Detailed Findings</div>', unsafe_allow_html=True)
    cm = {'pass': ('#2d6e2d', '#f0faf0'), 'fail': ('#a32020', '#fef5f5'), 'warn': ('#8a6200', '#fffbf0')}
    for c in checks:
        tc, bg = cm[c['status']]
        st.markdown(
            f'<div style="background:{bg};border-radius:8px;padding:9px 12px;margin-bottom:6px;">'
            f'<div style="font-size:12px;font-weight:600;color:{tc}">{SEMOJI[c["status"]]} {c["label"]}</div>'
            f'<div style="font-size:12px;color:#667080;margin-top:2px">{c.get("detail","")}</div>'
            f'</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    # Header
    st.markdown(
        '<div class="logo-pill"><span>askari</span>bank</div>'
        '<div class="header-sub">RECEIPT FRAUD DETECTOR · AI-POWERED</div>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Tesseract check
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        st.error(
            "**Tesseract not found.** Install it first:\n\n"
            "- **Windows:** [Download](https://github.com/UB-Mannheim/tesseract/wiki)\n"
            "- **macOS:** `brew install tesseract`\n"
            "- **Ubuntu:** `sudo apt install tesseract-ocr`")
        st.stop()

    # Load model + show status
    model, is_finetuned, real_idx, model_info = load_model()

    if is_finetuned:
        st.sidebar.markdown(
            f'<div class="model-badge-ok">🧠 Fine-tuned EfficientNet loaded</div><br>'
            f'<div style="font-size:11px;color:#667;margin-top:6px">{model_info}</div>',
            unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            f'<div class="model-badge-off">⚠️ Proxy mode — model not trained yet</div>'
            f'<div style="font-size:11px;color:#667;margin-top:8px">'
            f'To enable fine-tuned AI detection:<br>'
            f'<code>python train_model.py --data path/to/data</code>'
            f'</div>',
            unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Rule checks performed:**\n"
        "1. FUNDS TRANSFERRED heading\n"
        "2. Askari Bank logo/name\n"
        "3. Transaction ID label present\n"
        "4. Transaction ID: 12 digits, no letters\n"
        "5. Date: DD-Mon-YYYY HH:MM:SS AM/PM\n"
        "6. Transfer type: Within Askari / Interbank only\n"
        "7. Bank consistency (TO BANK matches type)\n"
        "8. No PKR/Rs currency symbol\n"
        "9. Numeric amount field"
    )

    # Upload
    st.markdown("**Upload a receipt image to verify**")
    uploaded = st.file_uploader(
        "Upload", type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed")

    if not uploaded:
        st.info("📤 Drag & drop or click to upload — PNG, JPG, WEBP supported.")
        st.stop()

    image = Image.open(io.BytesIO(uploaded.read()))
    st.image(image, caption="Uploaded Receipt", use_container_width=True)
    st.markdown("---")

    if st.button("🔍  Scan & Analyze Receipt", use_container_width=True, type="primary"):
        with st.status("Analyzing receipt…", expanded=True) as status:

            st.write("🧠 Running EfficientNet visual analysis...")
            try:
                eff = visual_score(image, model, is_finetuned, real_idx)
            except Exception as e:
                st.warning(f"Visual model error: {e}. Defaulting to 50%.")
                eff = {"visual_score": 50, "detail": "model error", "mode": "error"}

            st.write("🔤 Running Tesseract OCR...")
            try:
                ocr_text = run_ocr(image)
            except Exception as e:
                status.update(label="❌ OCR failed", state="error")
                st.error(f"Tesseract error: {e}")
                st.stop()

            st.write("📋 Running 9-point rule check...")
            ocr_result = run_ocr_checks(ocr_text)

            st.write("⚖️ Combining scores → final verdict...")
            result = combined_verdict(ocr_result, eff["visual_score"], is_finetuned)

            status.update(label="✅ Analysis complete!", state="complete", expanded=False)

        if not ocr_text.strip():
            st.warning("⚠️ Tesseract could not read any text. Try a higher-resolution image.")

        st.markdown("### Analysis Result")
        render_verdict(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_scores(result, is_finetuned, eff["detail"], model_info)
        st.markdown("<br>", unsafe_allow_html=True)
        render_checks(result['checks'])
        st.markdown("<br>", unsafe_allow_html=True)
        render_findings(result['checks'])

        with st.expander("📄 Raw OCR Text", expanded=False):
            st.markdown(
                f'<div class="ocr-box">{ocr_text.strip() or "(no text extracted)"}</div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#9aa3b0;text-align:center;line-height:1.6;">'
        '100% local · No API key · '
        '<strong style="color:#7b52ab">EfficientNet B0</strong> + '
        '<strong style="color:#1a6fc4">Tesseract OCR</strong> + '
        '<strong style="color:#2d8a2d">9-Point Rule Engine</strong>'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
