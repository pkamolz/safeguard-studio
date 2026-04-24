# Safeguard Studio

Applied AI safety research portfolio investigating methods for detecting misuse, evaluating model robustness, and analyzing agentic safety risks in large language models.

## Project Overview

1. **Abuse/Misuse Classifier** — Multi-class classifier for detecting harmful content in conversational AI contexts, with production-oriented threshold analysis and error characterization.
2. **Red Team Evaluation Suite** — Systematic framework for testing LLM safeguard robustness against adversarial prompts, including over-refusal measurement.
3. **Evaluation Methodology & Usage Analysis** — Comparative study of LLM eval frameworks, applied to large-scale conversation data for abuse pattern detection.
4. **Agentic Safety Analysis** — Practitioner-informed threat model for agentic AI systems.

## Setup

1. Create a GitHub Codespace from this repository (4-core, 16GB+ recommended)
2. Dependencies install automatically via `.devcontainer/postCreateCommand`
3. Download datasets: `python scripts/download_data.py`
4. Configure W&B: `wandb login`

## Progress

- [ ] Phase 1: Abuse/Misuse Classifier (Weeks 1-3)
- [ ] Phase 2: Red Team Eval Suite (Weeks 3-5)
- [ ] Phase 3: Evals & Usage Analysis (Weeks 5-7)
- [ ] Phase 4: Agentic Safety Analysis (Weeks 7-8)
