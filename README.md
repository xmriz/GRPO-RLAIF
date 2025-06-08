# 📘 Question Answering with RLAIF using GRPO

Sistem ini mengembangkan arsitektur *Question Answering* (QA) berbasis **Large Language Model (LLM)** untuk menjawab pertanyaan terkait regulasi keuangan di Indonesia, khususnya regulasi dari **Otoritas Jasa Keuangan (OJK)**. Model dilatih menggunakan pendekatan **Reinforcement Learning from AI Feedback (RLAIF)**, dengan fokus pada metode **Group Relative Policy Optimization (GRPO)**.

## 🚀 Proyek Ini Mencakup

* **Evaluasi Model Baseline** (pretrained LLM)
* **Supervised Fine-Tuning (SFT)** dengan data QA regulasi OJK
* **Reinforcement Learning GRPO** (dengan atau tanpa SFT)
* **Evaluasi Multi-metrik**: Exact Match, ROUGE, BLEU, METEOR
* **Eksperimen Terstandar** untuk semua model (Meta-Llama-3.1-8B, Aya-23-8B, SeaLLMs-v3-7B, dll.)

---

## 📂 Struktur Direktori

```
.
├── base-evaluation/              # Evaluasi model baseline (inference, EM, ROUGE, dll)
├── datasets/                     # Dataset prompt-completion QA berbasis regulasi OJK
├── download_model/              # Script untuk mengunduh model LLM
├── grpo/                         # GRPO dengan SFT (SFT-GRPO scenario)
├── grpo_only/                   # GRPO langsung tanpa SFT
├── sft/                          # Supervised Fine-Tuning (SFT-only scenario)
```

---

## 🤪 Evaluasi Eksperimen

| Skenario     | EM        | R-1       | R-2       | R-L       | BLEU      | METEOR    | Inference Time |
| ------------ | --------- | --------- | --------- | --------- | --------- | --------- | -------------- |
| Baseline     | 0.109     | 0.493     | 0.416     | 0.468     | 0.169     | 0.599     | 191.7 sec      |
| SFT-Only     | 0.227     | 0.672     | 0.593     | 0.659     | 0.339     | 0.726     | 197.2 sec      |
| GRPO-Only    | 0.121     | 0.499     | 0.423     | 0.476     | 0.174     | 0.599     | 197.2 sec      |
| **SFT-GRPO** | **0.229** | **0.660** | **0.582** | **0.647** | **0.314** | **0.725** | **197.5 sec**  |

📌 **Catatan**: Evaluasi dilakukan pada model **SeaLLMs-v3-7B**, menggunakan data uji 10% yang tidak digunakan saat pelatihan.

---

## ⚙️ Cara Menjalankan

### 1. Evaluasi Baseline Model

```bash
cd base-evaluation
python evaluate_base.py
```

### 2. Supervised Fine-Tuning (SFT)

```bash
cd sft
accelerate launch train_sft.py
```

### 3. Fine-Tuned Model Evaluation

```bash
cd sft
python evaluate_sft.py
```

### 4. GRPO Training

```bash
cd grpo
accelerate launch train_grpo.py
```

### 5. GRPO-Only Training

```bash
cd grpo_only
accelerate launch train_grpo.py
```

---

## 📊 Dataset

* **Format**: `prompt-completion` pairs (JSONL)
* **Bahasa**: Bahasa Indonesia
* **Domain**: Regulasi keuangan (dokumen OJK)
* **Ukuran**: ±5.000 data pelatihan dan 500 data uji

---

## 🔧 Konfigurasi Sistem

* **Model**: Meta-Llama-3.1-8B, Aya-23-8B, SeaLLMs-v3-7B, SEA-LION-v3-8B, Sahabat-AI-8B
* **Hardware**: Dual GPU NVIDIA A100 40GB
* **Quantization**: 4-bit (NF4, bfloat16)
* **Optimizer**: `paged_adamw_32bit` untuk SFT, GRPOConfig untuk GRPO
* **Training**:

  * SFT → 1 Epoch, LR = 1e-4
  * GRPO → 1 Epoch, LR = 1e-6, loss = `dr_grpo`

---

## 📊 Metrik Evaluasi

* **EM (Exact Match)**: Jawaban identik
* **ROUGE-1/2/L**: Overlap kata dan subsekuensi
* **BLEU**: Kemiripan n-gram
* **METEOR**: Penilaian semantik dan sinonim
* **Inference Time**: Waktu rata-rata per batch evaluasi

---

## 📄 Referensi

* Laporan Studi Mandiri (PDF)
* GRPO (Shao et al., 2024)
* DeepSeekMath & SFT+RL pipelines
* HuggingFace `trl`, `peft`, dan `accelerate`

---

## 👤 Penulis

**Ahmad Rizki**
Program Studi Teknik Informatika
Institut Teknologi Bandung – 2025
