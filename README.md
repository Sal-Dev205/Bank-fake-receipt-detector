# 🏦 Askari Bank — Receipt Fraud Detector

An AI-powered receipt authentication system that detects fraudulent Askari Bank fund transfer receipts using a combination of deep learning and rule-based analysis.

---

## 🧠 How It Works

The system uses two layers of analysis combined into a single fraud score:

### Layer 1 — EfficientNet B0 (Deep Learning)
A fine-tuned EfficientNet B0 convolutional neural network, pretrained on ImageNet and fine-tuned on a custom dataset of real and fake Askari Bank receipts. It learns to distinguish authentic WhatsApp receipt screenshots from AI-generated or digitally fabricated counterfeits based on visual and textural features.

- **Fine-tuned mode:** 45% weight in final score
- **Proxy mode** (if model not trained): falls back to softmax entropy analysis

### Layer 2 — 9-Point Rule Engine (OCR + Pattern Matching)
Tesseract OCR extracts text from the uploaded receipt image, which is then validated against 9 rules derived from real Askari Bank receipts:

| # | Check | Rule |
|---|-------|------|
| 1 | Bank heading | Must say "FUNDS TRANSFERRED" |
| 2 | Bank name | Must contain "Askari Bank" |
| 3 | Transaction ID label | Must have "Transaction reference ID #" |
| 4 | Transaction ID digits | Must be exactly 12 consecutive digits — no letters or symbols |
| 5 | Date format | Must be DD-Mon-YYYY HH:MM:SS AM/PM (12-hour, named month) |
| 6 | Transfer type | Only "Within Askari" or "Interbank" accepted |
| 7 | Bank consistency | Within Askari → beneficiary is Askari; Interbank → valid SBP bank |
| 8 | Currency symbol | No PKR or Rs symbol allowed |
| 9 | Amount field | Must contain a valid numeric amount |

### Final Verdict
| Score | Verdict |
|-------|---------|
| 85–100% | ✅ Authentic |
| 40–84% | ⚠️ Suspicious |
| 0–39% | ❌ Fraudulent |

---

## 🖥️ Screenshots

> Upload a receipt → get an instant verdict with a detailed breakdown of all 9 checks.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- Tesseract OCR installed on your system

**Install Tesseract:**
```bash
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# macOS:
brew install tesseract
# Ubuntu:
sudo apt install tesseract-ocr
```

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/askari-receipt-detector.git
cd askari-receipt-detector
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements_free.txt
```

### 4. Train the model *(run once)*
Prepare a `data/` folder with this structure:
```
data/
  real/    ← authentic Askari receipt images
  Fake/    ← fake or AI-generated receipt images
```
Then run:
```bash
python train_model.py --data data
```
This saves `askari_efficientnet.pth` in your project folder. Training takes 2–3 minutes.

### 5. Launch the app
```bash
streamlit run askari_receipt_detector_free.py
```
Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
askari-receipt-detector/
│
├── askari_receipt_detector_free.py   # Main Streamlit app
├── train_model.py                    # EfficientNet fine-tuning script
├── requirements_free.txt             # Python dependencies
├── README.md                         # This file
│
├── data/                             # Training dataset (not pushed to GitHub)
│   ├── real/
│   └── Fake/
│
└── askari_efficientnet.pth           # Trained model weights (generated after training)
```

---

## 🛠️ Built With

| Tool | Purpose |
|------|---------|
| [Python](https://python.org) | Core language |
| [Streamlit](https://streamlit.io) | Web UI framework |
| [PyTorch](https://pytorch.org) | Deep learning framework |
| [TorchVision](https://pytorch.org/vision) | EfficientNet B0 model |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | Text extraction from images |
| [Pillow](https://python-pillow.org) | Image preprocessing |

---

## 📌 Notes

- The `data/` folder and `askari_efficientnet.pth` are excluded from this repository via `.gitignore` due to size and privacy.
- The app works without the trained model (proxy mode) but fine-tuning significantly improves accuracy.
- Tesseract must be installed separately as a system binary — it is not a Python package.

---

## 👤 Author

**Salar Awais**  
AI Lab Project — 2026
