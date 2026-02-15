# BitDance for class-conditional image generation on ImageNet

## Environment Setup
We recommend using the text-to-image generation conda environment. It is compatible with class-conditional image generation.

## Model Checkpoints 

Model | Params | Step-256px | FID  | Huggingface 
--- |:---:|:---:|:---:|:---:|
Autoencoder | 460M | - | - |[ae_d16c32.pt](https://huggingface.co/shallowdream204/BitDance-ImageNet/blob/main/ae_d16c32.pt)
BitDance-B-1X   | 242M | 256 | 1.68 | [BitDance_B_1X.pt](https://huggingface.co/shallowdream204/BitDance-ImageNet/blob/main/BitDance_B_1X.pt)
BitDance-B-4X   | 260M |  64 |1.69 | [BitDance_B_4X.pt](https://huggingface.co/shallowdream204/BitDance-ImageNet/blob/main/BitDance_B_4X.pt)
BitDance-B-16X   | 260M |  16 |1.91 | [BitDance_B_16X.pt](https://huggingface.co/shallowdream204/BitDance-ImageNet/blob/main/BitDance_B_16X.pt)
BitDance-L-1X  | 527M |  256 |1.31 | [BitDance_L_1X.pt](https://huggingface.co/shallowdream204/BitDance-ImageNet/blob/main/BitDance_L_1X.pt)
BitDance-H-1X   | 1.0B |  256 |1.24 | [BitDance_H_1X.pt](https://huggingface.co/shallowdream204/BitDance-ImageNet/blob/main/BitDance_H_1X.pt)

Run the following code to download all model checkpoints.

```python
from huggingface_hub import snapshot_download

save_dir = "models/BitDance-ImageNet"
repo_id = "shallowdream204/BitDance-ImageNet"
cache_dir = save_dir + "/cache"

snapshot_download(cache_dir=cache_dir,
  local_dir=save_dir,
  repo_id=repo_id,
  local_dir_use_symlinks=False,
  resume_download=True,
  allow_patterns=["*.pt","*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
```

## Evaluation
1️⃣ Sample 50,000 images and save to `.npz`.

BitDance-B-1X:
```shell
ckpt=models/BitDance-ImageNet/BitDance_B_1X.pt
result_path=results
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --nnodes=1 --master_port=12345 \
sample_ddp.py --model BitDance-B --latent-dim 32 --trained-vae $vae_ckpt --ckpt $ckpt --cfg-scale 3.2 \
--sample-dir $result_path --per-proc-batch-size 384 --to-npz --chunk-size 64
```

BitDance-B-4X:
```shell
ckpt=models/BitDance-ImageNet/BitDance_B_4X.pt
result_path=results
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --nnodes=1 --master_port=12345 \
sample_ddp_parallel.py --model BitDance-B --latent-dim 32 --trained-vae $vae_ckpt --ckpt $ckpt --cfg-scale 3.9 \
--sample-dir $result_path --per-proc-batch-size 384 --to-npz --parallel-num 4 --chunk-size 64
```

BitDance-B-16X:
```shell
ckpt=models/BitDance-ImageNet/BitDance_B_16X.pt
result_path=results
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --nnodes=1 --master_port=12345 \
sample_ddp_parallel.py --model BitDance-B --latent-dim 32 --trained-vae $vae_ckpt --ckpt $ckpt --cfg-scale 6.1 \
--sample-dir $result_path --per-proc-batch-size 384 --to-npz --parallel-num 16 --chunk-size 64
```

BitDance-L-1X:
```shell
ckpt=models/BitDance-ImageNet/BitDance_L_1X.pt
result_path=results
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --nnodes=1 --master_port=12345 \
sample_ddp.py --model BitDance-L --latent-dim 32 --trained-vae $vae_ckpt --ckpt $ckpt --cfg-scale 4.0 \
--sample-dir $result_path --per-proc-batch-size 352 --to-npz --chunk-size 48
```

BitDance-H-1X:
```shell
ckpt=models/BitDance-ImageNet/BitDance_H_1X.pt
result_path=results
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --nnodes=1 --master_port=12345 \
sample_ddp.py --model BitDance-H --latent-dim 32 --trained-vae $vae_ckpt --ckpt $ckpt --cfg-scale 4.55 \
--sample-dir $result_path --per-proc-batch-size 224 --to-npz --chunk-size 32
```

2️⃣ These scripts generate a `.npz` file which can be directly used with [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute FID, Inception Score and other metrics.

## Training
1️⃣ Download the ImageNet dataset from the [official website](http://image-net.org/download).


2️⃣ Start training for BitDance.

BitDance-B-1X:
```shell
data_path=/path/to/imagenet/train/
result_path=results_bitdance_b_1x
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --master_addr=... --node_rank=... --master_port=12345 --nnodes=... \
train.py --results-dir $result_path --data-path $data_path --image-size 256 \
--model BitDance-B --epochs 800 --down-size 16 --latent-dim 32 \
--lr 6e-4 --global-batch-size 1024 --trained-vae $vae_ckpt --ema 0.9999 --perturb-rate 0.1
```

BitDance-B-4X:
```shell
data_path=/path/to/imagenet/train/
result_path=results_bitdance_b_4x
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --master_addr=... --node_rank=... --master_port=12345 --nnodes=... \
train_parallel.py --results-dir $result_path --data-path $data_path --image-size 256 \
--model BitDance-B --epochs 800 --down-size 16 --latent-dim 32 \
--lr 6e-4 --global-batch-size 1024 --trained-vae $vae_ckpt --ema 0.9999 --perturb-rate 0.1 --parallel-num 4
```

BitDance-B-16X:
```shell
data_path=/path/to/imagenet/train/
result_path=results_bitdance_b_16x
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --master_addr=... --node_rank=... --master_port=12345 --nnodes=... \
train_parallel.py --results-dir $result_path --data-path $data_path --image-size 256 \
--model BitDance-B --epochs 800 --down-size 16 --latent-dim 32 \
--lr 6e-4 --global-batch-size 1024 --trained-vae $vae_ckpt --ema 0.9999 --perturb-rate 0.1 --parallel-num 16
```


BitDance-L-1X:
```shell
data_path=/path/to/imagenet/train/
result_path=results_bitdance_l_1x
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --master_addr=... --node_rank=... --master_port=12345 --nnodes=... \
train.py --results-dir $result_path --data-path $data_path --image-size 256 \
--model BitDance-L --epochs 800 --down-size 16 --latent-dim 32 \
--lr 6e-4 --global-batch-size 1024 --trained-vae $vae_ckpt --ema 0.9999 --perturb-rate 0.05
```

BitDance-H-1X:
```shell
data_path=/path/to/imagenet/train/
result_path=results_bitdance_h_1x
vae_ckpt=models/BitDance-ImageNet/ae_d16c32.pt

torchrun --nproc_per_node=8 --master_addr=... --node_rank=... --master_port=12345 --nnodes=... \
train.py --results-dir $result_path --data-path $data_path --image-size 256 \
--model BitDance-H --epochs 800 --down-size 16 --latent-dim 32 \
--lr 6e-4 --global-batch-size 1024 --trained-vae $vae_ckpt --ema 0.9999 --perturb-rate 0.05
```

We train BitDance on H100 GPUs with the following setups: 16×H100 for BitDance-B, 32×H100 for BitDance-L, and 64×H100 for BitDance-H.


## Acknowledgement
This code is based on [SphereAR](https://github.com/guolinke/SphereAR). We thank the authors for their awesome work.
