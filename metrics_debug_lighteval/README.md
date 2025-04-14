# Goal
The goal of this directory was to figure out why we get weird results for the GSM8K and the Math dataset with lighteval.

We solved it by using the following:

## Dependencies:
- **lighteval**: We got the latest version, which is not the same as the original one. Maybe we should switch to the original one. I need to check. The version before that was 0.6.0.
- **vllm**: v0.6.3 because it's the last version with torch 2.4
- **torch**: 2.4.0 Not sure why actually.
- **python**: 3.10

We need to test 3.11