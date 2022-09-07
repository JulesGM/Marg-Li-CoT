# Marg-Li-CoT
Note: Uses http://www.github.com/julesgm/general_utils/

## Steps
1. Generate default scratch pads with GPT3
2. Refine GPT2 with the dataset
3. Fine-tune GPT2 with marginal likelihood

### Friday Aug 12th:

- Caden: Generate dataset with GPT-3
    - Implement group decoding 
    - Come up with a good few shot prompt
    - Try to look into making sure that the answer is not included

- Jules:
    - Will work on step 2 or step 3.
            - Step 2 isn't anything. 
            - Step 3: similar, different schedule etc.

# Jules: 
    ---> Sentences are too long. 
        - No length penalties ? Length penalty is = 1, same as bart.
        - No early stopping ? Early stopping only affects the beam search, not greedy search. We still turned it on.
        - Can't predict EOS? Do we even predict EOS? 
            We're pretty sure that we're indeed *training* on EOS.
            Is it possible that the model is somehow deleting it?
            
        
