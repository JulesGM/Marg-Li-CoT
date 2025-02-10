# LightEval evaluation code.

Code to evaluate the checkpoints of the Open-Instruct training runs on Hendrycks MATH & GSM8K.

- Use `eval_plot.ipynb` to consume the outputs.
- Use `lighteval_gsm8k.sh` & `lighteval_math.sh` to do individual evals.
- Use `multi_gpu_lighteval_chain.py` to run evals on all checkpoints of a task at once. Not 100% sure that this works.