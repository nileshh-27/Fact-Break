Fact-BreakReal-Time Citation Hallucination Detection and Prevention in Retrieval-Augmented Generation (RAG)Fact-Break is an active, real-time mechanistic circuit breaker middleware designed to detect and prevent citation hallucinations in Retrieval-Augmented Generation (RAG) pipelines.By attaching lightweight PyTorch forward hooks into the internal layers of an autoregressive transformer, FACT-BREAK monitors activation coordination between the model's Multi-Head Attention (context reading) and Feed-Forward Networks (parametric recall). When pathway decoupling indicates an ungrounded citation, the system immediately trips a circuit breaker to halt token decoding before hallucinated text reaches the user.OverviewWhile RAG architectures reduce factual drift by injecting external document context into Large Language Model (LLM) prompts, enterprise systems remain vulnerable to citation hallucinations. In these scenarios, the LLM emits a claim and appends an explicit citation tag (such as [Doc-101]), but the cited document either lacks supporting evidence or directly contradicts the statement.Traditional evaluation methods rely on post-hoc verification, such as secondary LLM-as-a-judge evaluators, verbatim n-gram string matching, or vector embedding similarity checks. These methods operate offline after the full completion has been streamed to the user, introducing significant latency overhead and computational cost.FACT-BREAK shifts citation safety from post-hoc evaluation to active, inline prevention by probing the transformer's residual stream during autoregressive decoding.Key FeaturesMechanistic Activation Probing: Injects non-invasive PyTorch forward hooks into intermediate transformer layers (Layers 14–28 in Qwen2.5-14B) to read internal activations without memory cloning or weight modification.Dual-Pathway Tracking: Computes real-time metric alignment between Multi-Head Attention (MHA) context routing and Feed-Forward Network (FFN) parametric memory recall on every generated token.Dynamic Circuit Breaker Interception: Immediately halts generation upon detecting activation decoupling, truncating trailing ungrounded claims and triggering a deterministic fallback router.Zero Fine-Tuning Required: Functions purely as runtime middleware without requiring expensive fine-tuning or parameter updates on the underlying foundation model.Low Latency Overhead: In-flight hook execution adds less than 3% per-token latency overhead (0.6 ms/token), drastically outperforming secondary evaluator LLM calls.System Architecture[ Input Query + Retrieved Context ]
                 │
                 ▼
     [ Autoregressive LLM Engine ]
                 │
                 ├─────► [ PyTorch Forward Hooks ] (Residual Layers 14 to 28)
                 │                 │
                 │                 ▼
                 │      [ Risk Evaluation Engine ] (Computes CAS, BAS, PFS, PAS)
                 │                 │
                 │          R(t) > Tau_risk?
                 │          ├───┴───┐
                 │       NO │       │ YES
                 │          ▼       ▼
                 └──────► [ Pass ] [ TRIPPED: Injection of EOS Token ]
                                    │
                                    ▼
                           [ Safe Fallback Router ]
Diagnostic MetricsFACT-BREAK evaluates intermediate activation metrics during generation step $t$:Context Attention Score (CAS): Measures attention weight distribution assigned by intermediate attention heads to retrieved document tokens relative to generated prompt tokens.Base Activation Score (BAS): Measures the relative energy contribution of FFN activation magnitude to the total residual stream state.Parametric Fallback Score (PFS): Measures the $L_2$-norm ratio of FFN update vector magnitude relative to MHA output magnitude. Spikes in PFS indicate parametric memory override.Pathway Alignment Score (PAS): Computes the directional cosine similarity between MHA context updates and FFN residual deltas.The overall Risk Index $R(t)$ is evaluated per token step:$$R(t) = \alpha (1 - \text{CAS}(t)) + \beta \cdot \text{PFS}(t) + \delta \cdot \text{BAS}(t) + \gamma (1 - \text{PAS}(t))$$When $R(t)$ breaches the critical threshold $\tau_{\text{risk}} = 0.65$, the engine forcefully terminates generation and appends a safe retrieval alert.Repository Structure.
├── backend_server.py      # FastAPI GPU server with PyTorch forward hooks & Qwen2.5-14B engine
├── benchmark_client.py    # Automated local test runner, CSV generator, and visualizer script
├── requirements.txt       # Python package dependencies
├── paper/                 # IEEE conference research paper source and PDF documentation
└── README.md              # Technical overview and installation guide
Experimental Setup & BenchmarksThe system was evaluated using a remote FastAPI backend running Qwen2.5-14B-Instruct in 4-bit NF4 quantization on dual NVIDIA Tesla T4 GPUs (30 GB VRAM), executing against an enterprise financial dataset across $N=500$ evaluation runs.Empirical Benchmark SummaryQuery CategoryQuery Type SampleBaseline Hallucination %FACT-BREAK StatusPost-Interception Hallucination %GroundedQ3 Revenue & Debt Ratios0.0%PASSED0.0%PoisonedQ4 Dividend Payout per Share66.7%TRIPPED (Tok 18)0.0%Temporal BoundarySingle-Month vs 9-Month Cash Flow40.0%TRIPPED (Tok 14)0.0%Out-of-BoundsCorporate Executive Information100.0%TRIPPED (Tok 6)0.0%FACT-BREAK allowed grounded queries to pass with a low false-positive rate (1.2%) while achieving a 0.0% post-interception hallucination rate across poisoned, temporal boundary, and out-of-bounds queries.Setup and InstallationPrerequisitesPython 3.10+PyTorch 2.0+ with CUDA supportNVIDIA GPU with $\ge$ 16 GB VRAM (for 14B model inference in 4-bit)1. Backend Server Deployment (GPU Environment)Install the required dependencies:pip install torch transformers bitsandbytes accelerate fastapi uvicorn pyngrok nest_asyncio
Run the backend server script:python backend_server.py
2. Local Benchmark Client ExecutionInstall local analytics packages:pip install requests pandas matplotlib seaborn
Configure API_BASE_URL in benchmark_client.py with your active backend server address and execute:python benchmark_client.py
This will run the test suite, print evaluation logs, write fact_break_benchmark_results.csv, and render comparative performance charts.Project Metadata & AttributionAuthor: Nilesh Reddy K (Department of Computer Science & Engineering, Koneru Lakshmaiah Education Foundation)Project Supervisor: Dr. Sushama Rani Dutta (Department of Computer Science & Engineering, Koneru Lakshmaiah Education Foundation)LicenseCopyright (c) 2026 Nilesh Reddy Karri. All rights reserved.
Proprietary and confidential. Access and use permitted solely under explicit written/verbal authorization only. For any queries please raise an issue or contact the me via email.


## License
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#license)

Copyright (c) 2026 Nilesh Reddy Karri. All rights reserved.  
Proprietary and confidential.
Access and use permitted solely under explicit written/verbal authorization only.
For any queries please raise an issue or contact the me via email. 
