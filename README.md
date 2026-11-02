# FACT-BREAK
### Real-Time Citation Hallucination Detection and Prevention in Retrieval-Augmented Generation (RAG)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/Transformers-Qwen2.5--14B-purple" />
  <img src="https://img.shields.io/badge/Architecture-RAG-green" />
  <img src="https://img.shields.io/badge/Status-Research%20Project-blue" />
</p>

---

## Overview

**FACT-BREAK** is an active, real-time mechanistic circuit breaker middleware designed to detect and prevent citation hallucinations in Retrieval-Augmented Generation (RAG) pipelines.

By attaching lightweight PyTorch forward hooks to the internal layers of an autoregressive transformer, FACT-BREAK monitors activation coordination between the model's Multi-Head Attention (context reading) and Feed-Forward Networks (parametric recall).

When pathway decoupling indicates an ungrounded citation, the system immediately trips a circuit breaker to halt token decoding before hallucinated text reaches the user.

## Problem Statement

While RAG architectures reduce factual drift by injecting external document context into Large Language Model (LLM) prompts, enterprise systems remain vulnerable to **citation hallucinations**.

In these scenarios, an LLM generates a claim and appends an explicit citation tag (such as `[Doc-101]`), but the cited document either lacks supporting evidence or directly contradicts the generated statement.

Traditional evaluation methods rely on post-hoc verification, including:

- **LLM-as-a-Judge:** Secondary language models evaluate generated responses against retrieved documents.
- **Lexical Matching:** N-gram similarity metrics such as ROUGE and BLEU identify textual overlap.
- **Semantic Embedding Similarity:** Vector-based comparisons measure semantic alignment but can overlook numerical contradictions and logical inconsistencies.

These methods operate after generation, potentially introducing additional latency and computational costs.

FACT-BREAK shifts citation safety from post-hoc evaluation to active, inline intervention by probing transformer activation states during autoregressive decoding.

## Key Features

- **Mechanistic Activation Probing:** Attaches PyTorch forward hooks to intermediate transformer layers (Layers 14–28 in Qwen2.5-14B) to monitor internal activation states without modifying model weights.

- **Dual-Pathway Tracking:** Computes real-time alignment metrics between Multi-Head Attention (MHA) context routing and Feed-Forward Network (FFN) parametric recall.

- **Dynamic Circuit Breaker Interception:** Halts generation when activation metrics exceed a calibrated risk threshold, truncates trailing ungrounded claims, and triggers a fallback retrieval mechanism.

- **Zero Fine-Tuning Required:** Operates as runtime middleware without requiring additional fine-tuning or parameter updates to the underlying foundation model.

- **Low Latency Overhead:** Reports an additional inference latency of 0.6 ms/token (2.48%) through in-flight activation monitoring.

## System Architecture

```text
       [ Input Query + Retrieved Context ]
                       |
                       v
             [ Autoregressive LLM ]
                       |
                       v
             [ PyTorch Forward Hooks ]
              (Residual Layers 14–28)
                       |
                       v
              [ Activation Buffer ]
                       |
                       v
              [ Risk Evaluation Engine ]
                 (CAS, BAS, PFS, PAS)
                       |
                       v
                R(t) > Tau_risk?
                  /          \
                NO            YES
                |              |
                v              v
             [ PASS ]    [ CIRCUIT TRIPPED ]
                |         [ Inject EOS Token ]
                |              |
                |              v
                |       [ Safe Fallback Router ]
                |              |
                v              v
                 [ Response Pipeline ]
```

## Diagnostic Metrics

FACT-BREAK evaluates intermediate transformer activation metrics during generation step \(t\).

### 1. Context Attention Score (CAS)

Measures the attention weight distribution assigned by intermediate attention heads to retrieved document tokens relative to generated prompt tokens.

### 2. Base Activation Score (BAS)

Measures the relative energy contribution of FFN activation magnitude to the total residual stream state.

### 3. Parametric Fallback Score (PFS)

Measures the ratio of the L2-norm magnitude of the FFN update vector relative to the MHA output magnitude.

Higher PFS values indicate a greater contribution from parametric FFN updates relative to attention outputs.

### 4. Pathway Alignment Score (PAS)

Computes the directional cosine similarity between MHA context updates and FFN residual deltas.

A reduction in pathway alignment, combined with increases in PFS and BAS, indicates potential decoupling between contextual grounding and parametric recall.

### Composite Risk Index

The overall Risk Index \(R(t)\) is evaluated at every generation step:

\[
R(t) = \alpha(1-CAS(t)) + \beta PFS(t)
\]

\[
\quad + \delta BAS(t) + \gamma(1-PAS(t))
\]

When the calculated risk exceeds the critical threshold:

\[
R(t) > \tau_{\text{risk}} = 0.65
\]

The circuit breaker terminates generation and triggers the safe fallback retrieval mechanism.

## Repository Structure

```text
FACT-BREAK/
│
├── backend_server.py
│   └── FastAPI GPU server with PyTorch hooks
│       and Qwen2.5-14B inference engine
│
├── benchmark_client.py
│   └── Automated benchmark runner,
│       CSV generator, and visualization scripts
│
├── requirements.txt
│   └── Python package dependencies
│
├── paper/
│   └── IEEE conference research paper
│       and PDF documentation
│
└── README.md
    └── Project documentation
```

## Experimental Setup & Benchmarks

FACT-BREAK was evaluated using a remote FastAPI backend running `Qwen2.5-14B-Instruct` with 4-bit NF4 quantization on a dual NVIDIA Tesla T4 GPU cloud setup (30 GB total VRAM).

The evaluation used an enterprise financial benchmark dataset comprising 10 distinct query profiles across 500 generation runs.

### Experimental Configuration

| Parameter | Configuration |
|---|---|
| LLM | Qwen2.5-14B-Instruct |
| Quantization | 4-bit NF4 |
| GPU Environment | Dual NVIDIA Tesla T4 |
| Total VRAM | 30 GB |
| Backend | FastAPI |
| Inference Framework | PyTorch / Transformers |
| Benchmark Profiles | 10 |
| Generation Runs | 500 |
| Runs per Profile | 50 |
| Decoding Strategy | Greedy Decoding |
| Temperature | 0 |

### Empirical Benchmark Results

| Query Category | Query Profile | Baseline Hallucination | FACT-BREAK Status | Post-Interception Hallucination |
|---|---|---:|---|---:|
| Grounded | Q3 Revenue & Debt Ratios | 0.0% | PASSED | 0.0% |
| Poisoned | Q4 Dividend Payout per Share | 66.7% | TRIPPED (Token 18) | 0.0% |
| Temporal Boundary | Single-Month vs 9-Month Cash Flow | 40.0% | TRIPPED (Token 14) | 0.0% |
| Out-of-Bounds | Corporate Executive Information | 100.0% | TRIPPED (Token 6) | 0.0% |

### Benchmark Observations

- **Grounded Queries:** FACT-BREAK allows context-supported generation to proceed while maintaining a reported false-positive rate of 1.2%.

- **Poisoned Queries:** The system identifies potential activation anomalies and interrupts generation before unsupported financial claims are completed.

- **Temporal Boundary Cases:** FACT-BREAK detects potential pathway decoupling when the model attempts to infer single-month financial values from aggregate reporting periods.

- **Out-of-Bounds Queries:** The system identifies requests for information absent from retrieved context and triggers the fallback mechanism.

**Reported Results:**

- 0.0% post-interception hallucination rate.
- 2.48% additional inference latency.
- 0.6 ms/token additional latency.
- 1.2% false-positive rate.

## Latency & Performance

The following measurements were reported across 500 evaluation runs.

| Performance Metric | Baseline | FACT-BREAK |
|---|---:|---:|
| Generation Latency | 24.2 ms/token | 24.8 ms/token |
| Generation Throughput | 41.3 tokens/sec | 40.3 tokens/sec |
| Additional Latency | — | 0.6 ms/token |
| Computational Overhead | — | 2.48% |

FACT-BREAK introduces runtime activation monitoring without requiring a secondary LLM evaluator or modifications to the underlying model weights.

## Installation & Setup

### Prerequisites

- Python 3.10 or later
- PyTorch 2.0 or later
- CUDA-compatible NVIDIA GPU
- NVIDIA GPU environment with sufficient VRAM for quantized 14B model inference
- Python package manager (`pip`)

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FACT-BREAK
```

### 2. Backend Server Deployment (GPU Environment)

Install the required dependencies:

```bash
pip install torch transformers bitsandbytes accelerate fastapi uvicorn pyngrok nest_asyncio
```

Start the backend server:

```bash
python backend_server.py
```

The backend initializes the quantized Qwen2.5-14B inference engine and exposes FastAPI endpoints for generation requests.

### 3. Local Benchmark Client Execution

Install the analytics and client dependencies:

```bash
pip install requests pandas matplotlib seaborn
```

Configure `API_BASE_URL` in `benchmark_client.py` with your active backend server address.

Execute the benchmark client:

```bash
python benchmark_client.py
```

The benchmark client generates:

- Benchmark execution logs.
- `fact_break_benchmark_results.csv`.
- Comparative performance charts.
- Evaluation summaries.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| LLM | Qwen2.5-14B-Instruct |
| Model Quantization | BitsAndBytes (NF4) |
| Model Integration | Hugging Face Transformers |
| Backend API | FastAPI |
| GPU Acceleration | CUDA |
| Benchmarking | Pandas |
| Data Visualization | Matplotlib, Seaborn |
| API Tunneling | ngrok |

## Future Scope

Potential extensions to FACT-BREAK include:

- Extending activation monitoring to multimodal RAG architectures.
- Supporting additional transformer architectures and foundation models.
- Improving adaptive risk threshold calibration.
- Exploring more granular attention and FFN activation attribution.
- Integrating with autonomous AI agents and tool-execution pipelines.
- Evaluating detection accuracy across larger and more diverse benchmark datasets.

## Project Metadata & Attribution

**Author:** Nilesh Reddy K  
Department of Computer Science & Engineering  
Koneru Lakshmaiah Education Foundation (KL University)

**Project Supervisor:** Dr. Sushama Rani Dutta  
Department of Computer Science & Engineering  
Koneru Lakshmaiah Education Foundation (KL University)

## Research Publication

The IEEE-style research paper and supporting documentation are available in the [`paper/`](paper/) directory.

## License & Usage Restrictions

Copyright (c) 2026 Nilesh Reddy Karri. All rights reserved.

FACT-BREAK is proprietary research software. Access, reproduction, modification, and redistribution are permitted only with explicit authorization from the copyright holder.

For permissions or licensing enquiries, please contact the author.

---

<p align="center">
  <b>FACT-BREAK — Detect. Interrupt. Ground.</b>
  <br/>
  Mechanistic Runtime Protection for Retrieval-Augmented Generation.
</p>
