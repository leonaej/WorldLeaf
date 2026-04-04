# WorldLeaf RL Agent: Evaluation Results





## Final Test Results vs RAG Baseline

| Metric | RAG Baseline | RL Agent | Delta |
|--------|-------------|----------|-------|
| **Overall Hit@1** | 49.12% | **61.76%** | **+12.64%** |
| Single-hop Hit@1 | 89.60% | 71.60% | -18.00% |
| Multi-hop Hit@1 | 38.16% | **45.63%** | **+7.47%** |

---

## Error Analysis

| Error Type | RAG Baseline | RL Agent | Delta |
|------------|-------------|----------|-------|
| HIT | 49.12% | **61.76%** | +12.64% |
| TYPE1 (traversal miss) | 10.50% | **5.51%** | -4.99% |
| TYPE2 (data gap) | 40.30% | **31.99%** | -8.31% |
| TYPE3 (wrong info) | 0.10% | 0.74% | +0.64% |



The Results Screenshot:

![Comparison](understanding_the_flow/screenshot_images_understanding/comparison.png)