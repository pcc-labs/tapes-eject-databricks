# tapes-eject-databricks

Paper labels your agent sessions. This demo takes those labels to Databricks, where they become datasets, evals, and a fine-tuned model.

- Act 1: capture and label sessions in Paper
- Act 2: sync the labeled sessions into Unity Catalog and an MLflow evaluation dataset
- Act 3: supervised fine-tune (SFT) of Qwen3-4B, using the sessions the labels selected
- Act 4: evaluate the base and fine-tuned models against `golden` and `regression` cases in MLflow

Design: [docs/specs/2026-09-29-databricks-labels-design.md](docs/specs/2026-09-29-databricks-labels-design.md)

Status: design only. No code yet.
