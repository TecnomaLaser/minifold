# MiniFold

MiniFold if a fast model for single chain protein structure prediction. Built using the same protein language model as ESMFold, it achieves considerable speedups (up to 10-20x) and memory savings (up to 10x) at little to no cost in performance.

[Paper Link](https://openreview.net/pdf?id=1p9hQTbjgo)

## Installation

Then install minifold, clone the repository and install it with:

`pip install .`

> We recommend installing in a fresh python environment.

## Inference

To run prediction, use the following:

`python predict.py example.fasta --out_dir PATH --cache PATH`

> The fasta header will be used to name the output file, set them wisely!

Options are available:

- `kernels`: uses our custom triton kernels
- `compile`: use torch.compile with dynamic shapes enabled
- `model_size`: one of `12L` or `48L` (default) 
- `token_per_batch`: maximum number of tokens that fit in your GPU, by default 2048
- `lm_dtype`: weight dtype for the frozen ESM-2 encoder, one of `fp16` (default), `bf16`, `fp32`
- `offload_lm`: stream the ESM-2 layers from host RAM one at a time
- `stream_attn`: project ESM attention maps as they are produced instead of stacking them
- `device`: `auto` (default), `cuda` or `cpu`

## Running on a small GPU

This fork makes MiniFold fit on consumer cards without retraining. Upstream
instantiates the 2.84e9-parameter ESM-2 encoder at torch's default dtype, so the
fp16 weights FAIR distributes are upcast to **fp32 = 10.58 GiB** before the model
ever reaches the GPU. Nothing needs those fp32 masters: the forward pass already
runs under `torch.autocast(bfloat16)` and the encoder is frozen at inference.

`--lm_dtype fp16` (the new default) builds the encoder directly in the dtype the
weights ship in, halving both VRAM and host RAM. `--offload_lm` additionally keeps
the 36 transformer layers in pinned host RAM and streams them in one at a time,
leaving only ~0.28 GiB of weights resident.

Measured on an RTX 5070 Laptop (7.53 GiB usable), 48L checkpoint, 3 recycles,
single sequence. Peak VRAM in GiB:

| residues | upstream fp32 | `--lm_dtype fp16` | `+ --offload_lm` |
|---|---|---|---|
| 128  | OOM at load | 5.72 | 0.49 |
| 384  | OOM at load | 6.09 | 0.91 |
| 512  | OOM at load | OOM  | 1.27 |
| 1024 | OOM at load | OOM  | 3.83 |
| 1280 | OOM at load | OOM  | 5.80 |
| 1536 | OOM at load | OOM  | OOM  |

Offloading costs about 0.7 s per prediction, so prefer the resident mode for
chains under ~450 residues. For long chains also set:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

without which allocator fragmentation caps inference at 1024 residues rather
than 1280. Beyond ~1280 residues the folding trunk's fp32 triangular updates,
not the encoder, are what bounds the run.

### Accuracy

fp16 is bit-exact with the released ESM-2 weights, so it is the faithful
low-memory choice; bf16 keeps the fp32 exponent range but rounds the mantissa.
Against an fp32 reference of the same checkpoint, `example.fasta`:

| target | `--lm_dtype fp16` | `--lm_dtype bf16` |
|---|---|---|
| 7L8N_A (151 aa) | 0.044 A CA-RMSD | 0.047 A |
| 7W74_A (238 aa) | 0.054 A CA-RMSD | 0.324 A |

Mean pLDDT agrees with the reference to within 0.07 (fp16). Validated on two
proteins only; `--lm_dtype bf16` remains available as the conservative fallback
if a sequence ever overflows fp16's narrower range.

## Training

We train the model using AFDB proteins filtered to > 70 global plddt and selected for diversity using uniref30 as initial list, pre-filtering. You may use the provided train.py script and the YAML configs under `configs`.

## Kernels

We developed two triton kernels for this work. You can find them [here](https://github.com/jwohlwend/minifold/tree/main/minifold/model/kernels).

## Cite

```
@article{
  wohlwend2025minifold,
  title={MiniFold: Simple, Fast, and Accurate Protein Structure Prediction},
  author={Jeremy Wohlwend and Mateo Reveiz and Matt McPartlon and Axel Feldmann and Wengong Jin and Regina Barzilay},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=1p9hQTbjgo},
  note={Featured Certification}
}
```
