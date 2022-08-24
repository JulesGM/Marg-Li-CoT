## What is done
A lot of stuff. 
 - Text data mode 
 - State load / state init & checkpointing
 - Pytorch lightning model
 - Fix the way the input and scratchpad are put together
 - Generation based validation
 - Pre-tokenized hdf5 mode

## To do
### High priority
 - Test Horovod
 - Test checkpointing / state resuming
 - Look at the data. Specifically, what the training_step sees, and what the generator sees, and what the validation sees.
 - Verify generation based validation

### Low priority
 - LR scheduler
 - Pre-tokenized data mode
 - Logits data mode

 ## Random thoughts
 Aug 15th:
    Should the refining and the fine-tuning be in the same script? 
     - Saving multiple checkpoints and then launching the job would be a lot more simple, would maybe keep the complexity of each script down.
     - Putting both in the same script would reduce repetition? 
     - Putting both in the same script would would reduce the need for an overarching coordination script.