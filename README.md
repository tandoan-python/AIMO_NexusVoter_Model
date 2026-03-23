# 🏆 Nexus-Voter: AI Mathematical Olympiad (AIMO) Progress Prize 3

Welcome to the Nexus-Voter repository! This project contains the complete inference pipeline used for the **Kaggle AI Mathematical Olympiad - Progress Prize 3.**

The goal of this competition is to build open-weight AI models capable of solving complex, Olympiad-level mathematical problems (written in LaTeX) with absolute precision.

## 🧠 Architecture Overview

**Nexus-Voter** employs a **Neuro-Symbolic** approach combined with **Self-Consistency Majority Voting**. LLMs are known to hallucinate mid-step and struggle with arithmetic. Instead of letting the AI guess the final answer, we turn the AI into an "Agentic Coder".

**Core Components:**

1. The Brain (LLM): Uses `Qwen2.5-Math-7B-Instruct-AWQ` loaded via `vLLM` for high-throughput generation on Kaggle's limited T4x2 GPUs.

2. The Calculator (Python REPL): An isolated, timeout-protected sandbox. The LLM writes a Python script using `sympy/math`, executes it, and reads the output to correct its own logic.

3. The Judge (Majority Voting): Each problem is evaluated 15-30 times using different reasoning paths (Temperature > 0). The most frequently occurring valid integer answer is selected as the final submission, significantly reducing variance and boosting accuracy.

4. Kaggle API Integration: Fully integrated with Kaggle's `AIMO3InferenceServer` using the `polars` dataframe logic required by the competition gateway.

## 🚀 How It Works (The Loop)
```
[Problem] -> LLM Thinks -> LLM Writes Code -> REPL Executes -> LLM Observes Output
   ^                                                                 |
   |_____________________ Iterates up to 5 times ____________________|

```
*If it fails after 5 iterations, a Fallback Mechanism forces the LLM to output its best direct guess.*

## ⚙️ Setup & Usage on Kaggle

To run this notebook successfully on the Kaggle platform without Internet access (Submission requirement):

1. Hardware Setup: - Select GPU T4x2 in your Kaggle Notebook accelerator settings.

2. Offline Model:

- Attach the `Qwen2.5-Math-7B-Instruct-AWQ` dataset to your notebook.

- Update `MODEL_PATH` in the code if your dataset path differs.

3. Offline Libraries (vllm):

You must install vllm via an offline dataset containing the .whl files since internet is disabled during the final scoring run.

Example: `!pip install --no-index --find-links=/kaggle/input/vllm-wheels vllm`

4. Execution:

Simply Run All. The script will automatically communicate with Kaggle's hidden evaluation server via `aimo_3_inference_server`.

## 📜 License

This project is subject to the competition's open-source requirements. Released under the [CC-BY 4.0 / MIT License].
