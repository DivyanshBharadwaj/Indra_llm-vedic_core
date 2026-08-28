# Indra LLM — Vedic Core

**An experimental Indic-language language-model project by Divyansh Bharadwaj (Shashwat Mudgal).**

> **Project lineage:** development is publicly recorded in this repository from September 28, 2025, with an Indra model integration commit dated October 5, 2025.

## Status

This repository contains an earlier architectural generation of the Indra project. It is retained as part of the project's research history and should not be interpreted as the current Indra architecture or as a production-ready model.

The original Vedic Core architecture was an experimental decoder-only transformer exploring Sanskrit/Indic language modeling, Vedic-text specialization, and architectural mechanisms such as GQA, MoE, retrieval/memory components, and hierarchical reasoning. Evaluation showed that this direction was not the basis for the current model generation.

## Evolution of Indra

Indra is a continuing model project rather than a single fixed architecture.

### Generation 1 — Vedic Core

The original Indra Vedic Core explored a Vedic-aligned decoder-only transformer with specialized components for Indic and Sanskrit text. This architecture was ultimately abandoned after evaluation.

### Current Generation — Qwen3-derived Indra architecture

The subsequent Indra architecture was substantially redesigned around the Qwen3 transformer design, with modifications made for the Indra project and specifically structured so that compatible Qwen3 weights could be used to hot-start the model.

This is a successor architecture, not a claim that the original Vedic Core implementation and the current architecture are identical.

### Indra Lite — current from-scratch training

The current Indra Lite series is being trained **from scratch** rather than being released as a Qwen3 fine-tune. The Qwen3-compatible design was used to make hot-start experimentation possible, while the present from-scratch training is intended to establish Indra's own learned weights and behavior.

Indra Lite is still under active development and is not yet a finished or production model.

## Research direction

The broader Indra project focuses on language modeling for Indic languages, with particular interest in Sanskrit, Hindi, linguistic structure, grammar-aware modeling, efficient architectures, and evaluation of model behavior on Indic text.

Architectural decisions and training strategies are expected to evolve as experiments are evaluated. Earlier experiments are preserved where possible so that the development history remains reproducible and auditable.

## Provenance and development history

This repository is part of the public development record for Indra. GitHub's commit history records work by **DivyanshBharadwaj** beginning on **September 28, 2025**. An Indra-specific model integration commit, `Create indra_model.py`, is dated **October 5, 2025**.

The current repository is therefore an archival research record of an earlier Indra architecture, while the current Indra Lite work is a later architectural generation.

## Author

**Divyansh Bharadwaj (Shashwat Mudgal)**

GitHub: [@DivyanshBharadwaj](https://github.com/DivyanshBharadwaj)

## Citation

```bibtex
@software{bharadwaj_indra_2025,
  author = {Divyansh Bharadwaj (Shashwat Mudgal)},
  title = {Indra LLM — Vedic Core},
  year = {2025},
  url = {https://github.com/DivyanshBharadwaj/Indra_llm-vedic_core}
}
```

## Note on naming

"Indra" is used here as the name of the Indra model project and its successive architectural generations. The repository history documents the project's development and evolution independently of later model releases or similarly named projects.
