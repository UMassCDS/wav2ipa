# About fine_tuning_experiments
These take a wav2vec2.0 model originially fine-tuned on TIMIT and further fine-tune it on Buckeye.

## `hyperparam_tuning`
Vary model parameters like learning rates and batch size (using the same training data in each experiment) to establish a reasonable baseline.
Note: effective batch size = batch per device x gradient accumulation steps x num GPUs

Goals:
- Figure out which model parameters produce good performance
- Establish baseline for our model architecture on the Buckeye corpus
- Check for any warning signs that the model architecture may not be appropriate, like over/underfitting


Params to vary:
- Effective batch size: [64, 32] (achieve these by varying batch size per device, number of gpus and grad accumulation steps appropriately)
    - To complete training quickly, you can use 4 or 8 GPUs on Unity, but they have to be large to get enough VRAM.
    - Note: effective batch size = batch per device x gradient accumulation steps x num GPUs
- Learning rate: [6e-4, 3e-4, 9e-5]