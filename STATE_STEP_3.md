## To Do:
 - Basically, copy-paste most of step 2, then:
     - Change the training step to:
        1. Generate the scratchpads with group beam search
        2. Marginalize over them. How does that look like in practice?
           Both of the following can (and should) be computed with just one run of the model.
            2.1 Compute the probability of the answer conditioning on the scratchpad + question. Softmax that.
            2.2 Compute the probability of the scratchpad knowing the question. Softmax that?
         
     - Change the loss
     - Save the output tokens elsewhere

   - If we were to do the code in step 2:
      - What would be the same:
         - Dataset objects would be the same
         - The valid dataloader
         - The valid_step would be the same
         - Data preprocessing would be the same

      - What would be different:
         - We would need a new train dataloader
         - We would need a new training step
         - We would need a different config file