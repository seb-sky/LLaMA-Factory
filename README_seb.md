# Seb's notes

# Installation

Create env

```shell
python3 -m venv .venv
```

Activate env

```shell
source .venv/bin/activate
```

Install requirements

```shell
pip install -r requirements.txt
```

## single GPU examples

### AQLM

* download AQLM model
* requires dev transformers, not tested yet

### AWQ

* download AWQ model
* ```pip install autoawq```

### bitsandbytes

* note: meta-llama/Llama-2-7b-hf is gated, need HUGGING_FACE_HUB_TOKEN
* ```pip install "bitsandbytes>=0.39.0"```
* script contains `--quantization_bit 4`

### GPTQ

* download GPTQ model
* ```pip install "optimum>=1.12.0"```
* ```pip install auto-gptq```

## unsloth

```pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" xformers```

## tensorboard

```pip install tensorboard```

## flash-attn

```shell
pip install --upgrade wheel
pip install flash-attn
```

