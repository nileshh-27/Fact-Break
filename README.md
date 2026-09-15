# Fact-Break

Real-Time Citation Hallucination Detection and Prevention in Retrieval Augmented Generation (RAG) aka **Fact-Break** is an active, real-time mechanistic circuit breaker middleware designed to prevent citation hallucinations in pipelines. 


By attaching lightweight PyTorch forward hooks into the internal layers of an autoregressive transformer, FACT-BREAK monitors the coordination between the model's **Multi-Head Attention** (context reading) and **Feed-Forward Networks** (parametric recall). When pathway decoupling indicates an ungrounded citation, the system immediately trips a circuit breaker to halt token decoding before the hallucinated text reaches the user.






## License
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#license)

Copyright (c) 2026 Nilesh Reddy Karri. All rights reserved.  
Proprietary and confidential.
Access and use permitted solely under explicit written/verbal authorization only.
For any queries please raise an issue or contact the me via email. 
