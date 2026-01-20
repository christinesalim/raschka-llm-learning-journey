# Learning Concepts

This directory contains self-contained learning modules with explanations and runnable demos.

## Available Concepts

### 1. Transpose in Multi-Head Attention

Understanding `transpose(1, 2)` and why it's crucial for parallel processing of attention heads.

- **Theory**: [transpose_explanation.md](transpose_explanation.md)
- **Demo**: [transpose_demo.py](transpose_demo.py)

**Quick start:**
```bash
python transpose_demo.py
```

---

## How to Use This Directory

Each concept includes:
- 📝 **Explanation file** (`.md`) - Theory, diagrams, and key takeaways
- 🐍 **Demo script** (`.py`) - Runnable code to visualize the concept

Keep notes and demos together so you can:
1. Read the explanation
2. Run the demo to see it in action
3. Modify the demo to experiment

---

## Template for New Concepts

When adding a new concept:

1. Create files:
   - `concept_name_explanation.md` - Your notes
   - `concept_name_demo.py` - Runnable example

2. Add entry to this README

3. Reference from implementation:
   ```python
   # See notes/concepts/concept_name_explanation.md for details
   ```
