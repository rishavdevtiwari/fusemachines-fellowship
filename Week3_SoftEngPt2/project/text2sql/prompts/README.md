# Prompts directory

This directory is reserved for LLM prompt templates if you later swap the
rule-based decomposer for a prompt-chaining LLM approach (Option 2 of
Task 3).

Suggested layout when you fill it in:

```
prompts/
├── 01_decompose.txt   # NL question -> structured JSON
├── 02_generate.txt    # decomposition -> SQL
└── 03_repair.txt      # SQL + error -> fixed SQL
```

The current pipeline does NOT call any LLM, so no prompts are required.
