# Adaptive Confidence-Based Hybrid Fusion for Robust Cyberbullying Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-darkgreen)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official repository and implementation of the paper:**  
> *"Adaptive Confidence-Based Hybrid Fusion for Robust Cyberbullying Detection Using Machine Learning and Transformer Models"*  
> **Authors:** Manik Patra, Chanchal Patra, Sarah Fatema, Rajdeep Pal, Sadananda Banerjee, Anindita Panda, Debasis Giri  
> *(MAKAUT, IIT BHU Varanasi, IIEST Shibpur)*  
> 📧 Contact: [manikpatra409@gmail.com](mailto:manikpatra409@gmail.com)

---

## 📌 Executive Summary

Automatic cyberbullying and toxic content detection faces severe vulnerabilities in real-world social media environments due to **adversarial lexical obfuscation** (e.g., leetspeak substitution such as `h@te`, `st*pid`, `$tup!d`, vowel omission, and glyph injection). 

* **Classical Machine Learning** (Character $n$-grams) is resilient to typographical noise and spelling mutations but lacks contextual and semantic comprehension.
* **Pretrained Deep Transformers** (e.g., RoBERTa) offer superior linguistic representation but suffer significant tokenization fragmentation and uncertainty degradation under out-of-vocabulary evasion attacks.

To resolve this trade-off, this repository presents a novel **Reinforcement Learning-Gated Dual-Branch Hybrid Architecture**:
1. **Branch A (Character & Lexical Ensemble)**: 5-Fold Stratified Stacking ensemble combining Word TF-IDF ($1\text{--}2$ $n$-grams), Character TF-IDF ($3\text{--}5$ $n$-grams), and 9 domain-specific behavioral features, calibrated via Platt scaling.
2. **Branch B (Contextual Transformer Deep Head)**: Pretrained `roberta-base` contextual representations routed through an attention-mask-aware **1D CNN + Bidirectional LSTM** classification head, calibrated via learned temperature scaling ($T = 1.082$).
3. **Branch C (RL Contextual Gating Agent)**: A continuous Actor-Critic **Proximal Policy Optimization (PPO)** agent initialized via **Contextual Bandit behavior cloning**. The agent dynamically predicts instance-level fusion weights $w_{ml} \in [0, 1]$ based on an 8-dimensional state vector under an **asymmetric safety-critical reward function** penalizing False Negatives ($2\times$) twice as strictly as False Positives.

On a multi-source harmonized corpus of **46,000 comments**, the proposed PPO RL-Adaptive Fusion achieves **95.84% Accuracy**, **98.15% Precision**, **0.9580 Macro-F1**, and **0.9891 ROC-AUC**. Under **35% severe adversarial obfuscation**, it preserves **97.20% clean performance** ($\text{F1} = 0.9312$), outperforming static heuristic fusion by **+3.15% F1** with only **0.15 ms** gating latency overhead.

---

## 🌟 Key Highlights

- **🛡️ Adversarial Resilience**: Retains $97.20\%$ performance under heavy character perturbation ($35\%$ leetspeak noise), significantly surpassing standalone transformers ($\Delta \text{F1} = +0.0287$).
- **🤖 Reinforcement Learning Gating**: Dynamic instance-level model arbitration using PPO trained on an 8D contextual state (confidence margins, text length, obfuscation density, class priors).
- **⚖️ Asymmetric Victim-Centric Optimization**: Formulates an asymmetric reward structure ($\text{FN} = -2.0, \text{FP} = -1.0$) to prioritize victim protection without triggering excessive benign censorship.
- **⚡ Two-Tier Production Readiness**:
  - **Tier 1 (Sub-millisecond Edge)**: Classical Stacking ML ($0.48\text{ ms/sample}$, $13.62\text{ MB}$ footprint) for real-time live chat streams.
  - **Tier 2 (Adaptive GPU Escalation)**: PPO RL-Adaptive Fusion ($2.45\text{ ms/sample}$) for safety-critical edge cases and ambiguous content ($C_{ml} < 0.35$).
- **🌐 Cross-Dataset Transferability**: Evaluated across 3 distinct online platforms (AndrewMVD Twitter, MRMORJ Hate Speech, MOMO Cyberbully) with zero catastrophic accuracy loss.

---

## 📐 End-to-End System Architecture

```text
                                  RAW SOCIAL MEDIA TEXT
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │       Unified Text Preprocessing Pipeline     │
                    │  • ftfy Bad Character & HTML Entity Unescape  │
                    │  • Unicode NFKC Normalization                 │
                    │  • Emoji Demojization (e.g. :rage:, :skull:)  │
                    │  • Leetspeak Normalization (h@te -> hate)     │
                    │  • Contraction Expansion (don't -> do not)    │
                    │  • Punctuation / URL / Mention Filtering      │
                    │  • WordNet Lemmatization                      │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
   ┌─────────────────────────────────┐             ┌─────────────────────────────────┐
   │    BRANCH A: CLASSICAL ML       │             │   BRANCH B: DEEP TRANSFORMER    │
   │  • Word TF-IDF (1-2 n-grams)    │             │  • Pretrained RoBERTa-base      │
   │  • Char TF-IDF (3-5 n-grams)    │             │  • 1D CNN Feature Extractor     │
   │  • 9 Handcrafted Lexical Cues   │             │  • Bidirectional LSTM Head      │
   │  • Logistic Reg + LinSVC + XGB  │             │  • Attention-Masked Pooling     │
   │  • 5-Fold Stratified Stacking   │             │  • Temperature Scaling (T=1.08) │
   │  • Sigmoid (Platt) Calibration  │             │                                 │
   │    Output: P_ml, C_ml           │             │    Output: P_tr, C_tr           │
   └────────────────┬────────────────┘             └────────────────┬────────────────┘
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      BRANCH C: RL CONTEXTUAL GATING AGENT     │
                    │               8-Dimensional State s_t         │
                    │   [P_ml, C_ml, P_tr, C_tr, Obf, Len, Prior, 1]│
                    │                       │                       │
                    │        Contextual Bandit Warm-Start           │
                    │                       │                       │
                    │       Actor-Critic PPO Policy Network         │
                    │       Reward: FN = -2.0, FP = -1.0            │
                    │                       │                       │
                    │          Dynamic Weight w_ml ∈ [0, 1]         │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │     Adaptive Confidence-Based Fusion Layer    │
                    │       P_final = w_ml · P_ml + (1 - w_ml) · P_tr│
                    │         Optimal Threshold Decision τ* = 0.50  │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                          FINAL PREDICTION: [0: Non-Bullying | 1: Cyberbullying]
```

---

## 🔬 Mathematical Formulation

### 1. Confidence Margin Estimation
For both the classical stacking pipeline and the transformer pipeline, instance-level certainty is derived from calibrated posterior probabilities ($P \in [0, 1]$):
$$C_{ml} = 2 \cdot |P_{ml} - 0.5| \in [0, 1], \qquad C_{tr} = 2 \cdot |P_{tr} - 0.5| \in [0, 1]$$
A score near $1.0$ indicates decisive certainty, while $0.0$ signifies maximal epistemic ambiguity (neutral decision boundary).

### 2. 8-Dimensional State Representation ($s_t$)
At inference step $t$, the RL agent observes state vector $s_t \in \mathbb{R}^8$:
$$s_t = \begin{bmatrix} P_{ml}, & C_{ml}, & P_{tr}, & C_{tr}, & \text{Obf}(x), & \text{Len}(x), & \rho_{pos}, & 1.0 \end{bmatrix}^T$$
where:
* $\text{Obf}(x) \in [0, 1]$ denotes normalized leetspeak and special character density.
* $\text{Len}(x) \in [0, 1]$ denotes normalized character count relative to the 99th percentile ($P_{99} = 280$).
* $\rho_{pos}$ represents the training class prior balance ($0.50$).
* $1.0$ serves as a stationary policy bias coordinate.

### 3. Asymmetric Safety-Critical Reward Function
In the contextual cyberbullying moderation environment, letting abusive content slip through (False Negative) causes immediate victim distress, whereas false censorship (False Positive) is comparatively tolerable. The reward function is formalized as:
$$R(a, y) = \begin{cases} 
+1.0, & \text{if } \hat{y} = y = 1 \quad (\text{True Positive}) \\
+1.0, & \text{if } \hat{y} = y = 0 \quad (\text{True Negative}) \\
-1.0, & \text{if } \hat{y} = 1 \land y = 0 \quad (\text{False Positive}) \\
-2.0, & \text{if } \hat{y} = 0 \land y = 1 \quad (\text{False Negative, } 2\times \text{ penalty})
\end{cases} - \lambda_{lat} |w_{ml} - 0.5|$$
with $\lambda_{lat} = 0.02$ as a variance regularizer to prevent violent action oscillations.

### 4. Dynamic Hybrid Fusion
The combined probability of cyberbullying is given by:
$$P_{final} = w_{ml} \cdot P_{ml} + (1 - w_{ml}) \cdot P_{tr}$$
$$\hat{y} = \mathbb{I}(P_{final} \ge \tau^*), \quad \text{where } \tau^* = 0.50$$

---

## 📊 Experimental Results

All experiments are conducted on the harmonized held-out test partition ($N = 6,900$ samples) across all three data distributions.

### Main Benchmark Comparison

| Model / Architecture | Accuracy | Precision | Recall | $\text{F1}(\tau^*)$ | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical Stacking ML (5-Fold Calibrated) | 0.9510 | 0.9758 | 0.9249 | 0.9497 | 0.9510 | 0.9821 |
| Transformer CNN-BiLSTM (`roberta-base`) | 0.9533 | 0.9641 | **0.9417** | 0.9528 | 0.9533 | 0.9877 |
| Random Weight Fusion (Ablation) | 0.9482 | 0.9620 | 0.9285 | 0.9449 | 0.9481 | 0.9805 |
| Static Heuristic Fusion ($\gamma = 2.0$) | 0.9552 | 0.9792 | 0.9301 | 0.9541 | 0.9552 | 0.9870 |
| Contextual Bandit Fusion (Linear Q) | 0.9568 | 0.9785 | 0.9340 | 0.9558 | 0.9561 | 0.9882 |
| **PPO RL-Adaptive Fusion (Proposed)** | **0.9584** | **0.9815** | 0.9352 | **0.9577** | **0.9580** | **0.9891** |

> *Statistical Significance*: The McNemar test between Static Heuristic Fusion and PPO RL-Adaptive Fusion yields $\chi^2 = 12.84$ ($p < 0.001$), confirming statistically significant superiority.

---

### Comparison with State-of-the-Art (SOTA) Baselines

| Reference / Model | Methodology | Accuracy | F1-Score |
| :--- | :--- | :---: | :---: |
| Putri et al. | BiLSTM Baseline | 0.9150 | 0.9025 |
| Alkasassbeh et al. | Deep Learning Framework | 0.9420 | 0.9380 |
| Baseline Implementation | Single-source Undersampled | 0.8756 | 0.8725 |
| **Proposed Work** | **PPO RL-Adaptive Hybrid Fusion** | **0.9584** | **0.9577** |

---

### Adversarial Obfuscation Robustness Stress Test

Test samples were synthetically corrupted with leetspeak substitutions (`hate` $\to$ `h@te`, `bitch` $\to$ `b!tch`, `stupid` $\to$ `$t*pid`) at moderate ($15\%$) and severe ($35\%$) noise levels.

| Fusion Architecture | Clean F1 | Obfuscated F1 (15%) | Decay (15%) | Obfuscated F1 (35%) | Decay (35%) | **Retention Rate** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Classical Stacking ML | 0.9497 | 0.9115 | 4.03% | 0.8569 | 9.78% | 90.22% |
| Transformer CNN-BiLSTM | 0.9528 | 0.9298 | 2.41% | 0.9025 | 5.27% | 94.73% |
| Static Heuristic ($\gamma = 2.0$) | 0.9541 | 0.9326 | 2.25% | 0.8997 | 5.70% | 94.30% |
| Contextual Bandit (Linear Q) | 0.9558 | 0.9385 | 1.81% | 0.9180 | 3.98% | 96.02% |
| **PPO RL-Adaptive Fusion** | **0.9577** | **0.9450** | **1.33%** | **0.9312** | **2.80%** | **97.20%** |

---

### Cross-Dataset Generalization Breakdown

| Source Dataset Partition | Platform / Nature | Test Samples | Accuracy | Precision | Recall | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MOMO Cyberbully** | Social Media Comments | 4,174 | 0.9981 | 0.9942 | 1.0000 | 1.0000 |
| **MRMORJ Hate Speech** | Forum / Tweet Corpus | 987 | 0.9485 | 0.9821 | 0.9497 | 0.9754 |
| **AndrewMVD Cyberbullying**| Raw Twitter / Slang | 1,739 | 0.8682 | 0.9620 | 0.8416 | 0.9315 |
| **Combined Multi-Source Corpus** | **Total Held-Out Test Set** | **6,900** | **0.9584** | **0.9815** | **0.9352** | **0.9891** |

---

### Hardware Profiling & Computational Footprint

| Architecture / Tier | Latency (ms/sample) | Peak VRAM/RAM | Disk Footprint | Target Environment |
| :--- | :---: | :---: | :---: | :--- |
| **Classical Stacking ML** | **0.48 ms** | **0.10 MB** | **13.62 MB** | Edge / Real-time Live Streams |
| Transformer Pipeline | 1.81 ms | 358.10 MB | 477.03 MB | Cloud GPU Worker |
| Static Heuristic Baseline | 2.30 ms | 358.10 MB | 490.65 MB | Cloud GPU Worker |
| **PPO RL-Adaptive Fusion** | **2.45 ms** | **358.20 MB** | **492.20 MB** | **Production Hybrid Escalation** |

*Note: The RL decision gating layer introduces an overhead of only **0.15 ms** over the static heuristic baseline.*

---

### Systematic Ablation Study

| Ablation Configuration | Clean F1 | $\Delta\text{F1}$ | Obf. F1 (35%) | Retention Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Full Proposed PPO RL-Adaptive Fusion** | **0.9577** | — | **0.9312** | **97.20%** |
| w/o Behavior Cloning Warm-Start (Cold Start RL) | 0.9490 | $-0.0087$ | 0.8769 | 92.40% |
| w/o VecNormalize Reward Scaling | 0.9505 | $-0.0072$ | 0.8849 | 93.10% |
| w/o Asymmetric Penalty ($\text{FN} = \text{FP} = -1.0$) | 0.9522 | $-0.0055$ | 0.8998 | 94.50% |
| w/o Obfuscation State Feature ($s_t \in \mathbb{R}^7$) | 0.9535 | $-0.0042$ | 0.8931 | 93.80% |
| Contextual Bandit Baseline (Linear Q) | 0.9558 | $-0.0019$ | 0.9180 | 96.02% |
| Static Heuristic Fusion Baseline ($\gamma = 2.0$) | 0.9541 | $-0.0036$ | 0.8997 | 94.30% |

---

## 📂 Repository Structure

```plaintext
Adaptive-Confidence-Based-Hybrid-Fusion-for-Robust-Cyberbullying-Detection/
│
├── README.md                                                        # Comprehensive project documentation
├── Adaptive_Hybrid_Cyberbullying_Pipeline_RL_Enhanced.ipynb        # End-to-end executable Jupyter/Colab notebook
├── Paper_ID_366__Adaptive_Confidence_Based_Hybrid_Fusion_...pdf    # Full published research paper
│
├── src/                                                             # Core implementation modules (planned / modularized)
│   ├── preprocessing.py                                             # Text cleaning, demojization, leetspeak decoder
│   ├── classical_pipeline.py                                        # FeatureUnion + Stacking ML + Calibration
│   ├── transformer_head.py                                          # RoBERTa-base + CNN-BiLSTM + Attention Head
│   ├── rl_environment.py                                            # Gymnasium FusionEnv + 8D State Extraction
│   ├── ppo_agent.py                                                 # Contextual Bandit + Stable-Baselines3 PPO
│   └── fusion_inference.py                                          # Runtime dual-tier inference engine
│
└── figures/                                                         # Architectural charts, confusion matrices, ROC plots
```

---

## 🚀 Installation & Quickstart

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (e.g., Tesla T4, V100, A100, RTX 3060+) recommended

### 1. Clone the Repository
```bash
git clone https://github.com/patramanik/Adaptive-Confidence-Based-Hybrid-Fusion-for-Robust-Cyberbullying-Detection.git
cd Adaptive-Confidence-Based-Hybrid-Fusion-for-Robust-Cyberbullying-Detection
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```
*Or install directly via pip:*
```bash
pip install torch transformers datasets accelerate sentencepiece \
            emoji contractions ftfy xgboost scikit-learn nltk \
            pandas numpy psutil matplotlib seaborn gymnasium stable-baselines3 kagglehub
```

### 3. Run the Research Notebook
Launch the complete end-to-end pipeline in Jupyter or Google Colab:
```bash
jupyter notebook Adaptive_Hybrid_Cyberbullying_Pipeline_RL_Enhanced.ipynb
```
The notebook executes:
1. Multi-source dataset ingestion and label harmonization.
2. Unified text cleaning, emoji demojization, and leetspeak normalization.
3. 5-Fold Stratified Classical Stacking training & probability calibration.
4. Pretrained RoBERTa + CNN-BiLSTM deep learning head fine-tuning & temperature scaling.
5. Contextual Bandit initialization & 50,000-timestep PPO policy training.
6. Benchmark evaluations, adversarial robustness stress tests, and 6-panel analytical dashboards.

---

## 💡 Quick Inference Snippet

```python
from typing import Dict, Any
import numpy as np

# Sample text with adversarial leetspeak obfuscation
sample_text = "I will find you and make your life hell, you st*pid idiot! :rage:"

def predict_cyberbullying(
    text: str,
    stacking_pipeline,
    transformer_pipeline,
    ppo_policy,
    threshold: float = 0.50
) -> Dict[str, Any]:
    # 1. Classical ML confidence & probability
    p_ml, c_ml = stacking_pipeline.predict_proba_and_confidence(text)
    
    # 2. Fast edge filter: if classical confidence is very high, return immediately
    if c_ml > 0.85:
        return {
            "prediction": int(p_ml >= threshold),
            "probability": float(p_ml),
            "tier": "Tier 1 (Classical Edge)",
            "weight_ml": 1.0
        }
    
    # 3. Contextual Transformer probability
    p_tr, c_tr = transformer_pipeline.predict_proba_and_confidence(text)
    
    # 4. Construct 8D RL State
    obf_score = compute_obfuscation_density(text)
    norm_len = min(len(text) / 280.0, 1.0)
    state = np.array([p_ml, c_ml, p_tr, c_tr, obf_score, norm_len, 0.50, 1.0], dtype=np.float32)
    
    # 5. Predict dynamic weight via RL policy
    w_ml, _ = ppo_policy.predict(state, deterministic=True)
    w_ml = float(np.clip(w_ml, 0.0, 1.0))
    
    # 6. Adaptive Hybrid Fusion
    p_final = w_ml * p_ml + (1.0 - w_ml) * p_tr
    prediction = int(p_final >= threshold)
    
    return {
        "prediction": prediction,
        "probability": float(p_final),
        "tier": "Tier 2 (PPO RL-Adaptive GPU)",
        "weight_ml": w_ml,
        "weight_tr": 1.0 - w_ml
    }
```

---

## 🏭 Production Deployment Architecture

To balance **throughput**, **operational cost**, and **victim safety**, we recommend deploying the detector in a hierarchical two-tier configuration:

```text
Incoming Message
       │
       ▼
┌──────────────────────────────────────────────┐
│  Tier 1: Classical Stacking ML Edge Service   │  (Latency < 0.5 ms | CPU-friendly)
│  • Fast TF-IDF + Handcrafted Features        │
│  • Calibrated Confidence C_ml                │
└──────────────────────┬───────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
  Decisive Prediction             Ambiguous or Adversarial
  (C_ml ≥ 0.35)                   (C_ml < 0.35 or High Obfuscation)
       │                               │
       ▼                               ▼
  Instant Action                  ┌──────────────────────────────────────────────┐
  (Approve / Drop)                │ Tier 2: PPO RL-Adaptive GPU Escalation Service│
                                  │ • RoBERTa CNN-BiLSTM Contextual Head         │
                                  │ • PPO Policy Dynamic Weight Arbitration      │
                                  │ • Asymmetric Safety Decision τ* = 0.50       │
                                  └──────────────────────────────────────────────┘
```

---

## 📖 Citation

If you find this research, code, or methodology helpful in your academic work or commercial project, please cite:

```bibtex
@article{patra2024adaptive,
  title={Adaptive Confidence-Based Hybrid Fusion for Robust Cyberbullying Detection Using Machine Learning and Transformer Models},
  author={Patra, Manik and Patra, Chanchal and Fatema, Sarah and Pal, Rajdeep and Banerjee, Sadananda and Panda, Anindita and Giri, Debasis},
  journal={arXiv preprint},
  year={2024},
  institution={Maulana Abul Kalam Azad University of Technology, Indian Institute of Technology (BHU) Varanasi, and Indian Institute of Engineering Science and Technology, Shibpur}
}
```

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgments

- Datasets provided by [AndrewMVD](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification), [Thomas Davidson et al. (MRMORJ)](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset), and [MOMO](https://www.kaggle.com/datasets/momo12341234/cyberbully-detection-dataset).
- Pretrained weights and tokenizers provided by [Hugging Face](https://huggingface.co/roberta-base).
- RL framework provided by [Gymnasium (Farama Foundation)](https://gymnasium.farama.org/) and [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).
