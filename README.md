# Transformer-Based Language Model Surprisal Predicts Human Reading Times Best with About Two Billion Training Tokens

## Introduction
This is the code repository for the paper [Transformer-Based Language Model Surprisal Predicts Human Reading Times Best with About Two Billion Training Tokens](https://aclanthology.org/2023.findings-emnlp.128.pdf), including code for training autoregressive LMs using the [EleutherAI GPT-NeoX](https://github.com/EleutherAI/gpt-neox) library.

## Setup
1) Clone the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) repository, and revert it to the commit that was used in this work:
```
$ git clone https://github.com/EleutherAI/gpt-neox.git
$ cd gpt-neox
$ git reset --hard 038b01
```

2) Install the dependencies following the README of the resulting GPT-NeoX repository.

3) Copy all files under the `gpt-neox` directory of THIS repository to the GPT-NeoX repository. This will overwrite some files and add some new files.

4) Prepare the training data (10,000 batches from the Pile in this work) following the instructions outlined in the [EleutherAI Pythia](https://github.com/EleutherAI/pythia) repository. This should be a numpy array of size `(10000*1024, 2049)` saved in a `.bin` file under `gpt-neox/data` that can be loaded with the following line:
```
np.memmap(data_prefix+".bin", dtype=np.uint16, mode="r", order="C", shape=(10000*1024, 2049))
```

## LM Training
Running the command `python deepy.py train_slms.py CONFIG_FILE` (e.g. `python deepy.py train_slms.py configs/pythia-1-1-64-10k.yml`) under the GPT-NeoX repository will launch LM training.
Refer to the README of the GPT-NeoX repository for an explanation of each argument in the configuration.

Once training is complete, each checkpoint can be converted to the 'HuggingFace format' with the command `python tools/convert_sequential_to_hf.py --input_dir CHECKPOINT_DIR --config_file CONFIG_FILE --output_dir HF_MODEL_DIR` (e.g. `python tools/convert_sequential_to_hf.py --input_dir output_1_1_64_10k/global_step1000 --config_file configs/pythia-1-1-64-10k.yml --output_dir hf_models/output_1_1_64_10k/global_step1000`).

## LM Surprisal Calculation
Once the HuggingFace versions of checkpoints are in place, repositories like [this](https://github.com/byungdoh/llm_surprisal) can be used to calculate LM surprisal predictors.

## Pre-Trained Weights
Weights of models that were trained as part of Experiment 2 are too large to upload as part of this repository, but they can be shared upon request.

## Questions
For questions or concerns, please contact Byung-Doh Oh ([oh.531@osu.edu](mailto:oh.531@osu.edu)).
