# Data sources

Place the knowledge-base documents here, then run `python -m rag.ingest`.

## Required for the assignment
- **≥ 3 documents**, **≥ 2 PDFs**, **≥ 1 PDF with 400+ pages**.

## What this project expects

| File | Type | Notes |
|------|------|-------|
| `hiluxcare-support-policy.md` | Markdown | ✅ already included (company FAQ & policies) |
| `toyota-hilux-manual.pdf`     | PDF  | **You add this** — the official Toyota Hilux Owner's Manual (≈500+ pages, satisfies the 400+ requirement). Search "Toyota Hilux owner's manual pdf". |
| `toyota-warranty-guide.pdf`   | PDF  | **You add this** — a Toyota warranty & maintenance guide (~80 pages). |

Any `.pdf`, `.md`, or `.txt` file dropped in this folder is automatically picked
up by the ingestion script. PDF page numbers are preserved so the assistant can
cite "FileName, p. N".

> Large PDFs are tracked with Git LFS (see `.gitattributes`).
