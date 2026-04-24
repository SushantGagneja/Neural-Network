# MNIST Neural Network in x86-64 Assembly

This repository is a handwritten digit classifier built almost entirely in x86-64 assembly.

The network is a plain multilayer perceptron:

`784 -> 256 -> 128 -> 64 -> 10`

with:

- ReLU on the hidden layers
- softmax on the output layer
- mini-batch SGD
- learning-rate decay
- direct MNIST file loading from assembly

No framework is doing the interesting work here. The forward pass, backpropagation, weight updates, data loading, and evaluation loop are all implemented in assembly.

## Why this exists

The goal of the project was not just to "get MNIST working," but to understand what a neural network looks like when all the usual abstractions are stripped away.

That means dealing with things high-level ML code normally hides:

- manual tensor layout
- calling conventions and register discipline
- syscall behavior
- float math details
- debugging wrong accuracy caused by low-level bugs rather than model design

This repo went through exactly that kind of debugging. Two of the most important fixes were:

- `argmax.asm` was clobbering `rbx`, which corrupted evaluation loops
- `loader.asm` reused `rcx` across Linux `syscall`, which broke dataset loading and produced fake metrics

Those are the kinds of bugs this project was meant to surface.

## Architecture

| Layer | Shape | Activation |
|---|---:|---|
| Input | 784 | - |
| Hidden 1 | 256 | ReLU |
| Hidden 2 | 128 | ReLU |
| Hidden 3 | 64 | ReLU |
| Output | 10 | Softmax |

## Training setup

| Parameter | Value |
|---|---:|
| Epochs | 25 |
| Batch size | 64 |
| Initial learning rate | 0.005 |
| LR decay | 0.97 per epoch |
| Training samples used | 59,968 |
| Validation split | test samples 9000..9999 |
| Test set | 10,000 samples |
| Initialization | He-style scaling with deterministic LCG |

## Results

Final runs from this repo:

### Assembly

- Final validation accuracy: `94.0999%`
- Test accuracy: `0.9452`

### Python reference

- Final validation accuracy: `95.00%`
- Test accuracy: `0.9481`

The assembly model finishes within about `0.29%` test accuracy of the Python reference, which is close enough to give confidence that the implementation is behaving correctly.

## What’s in the repo

| File | Purpose |
|---|---|
| `main.asm` | Training loop, validation loop, final test pass |
| `forward_pass.asm` | Dense layer forward pass |
| `backward_pass.asm` | Backpropagation through all 4 layers |
| `optimizer.asm` | Parameter updates and gradient clearing |
| `memory.asm` | Activation and gradient buffers |
| `weights.asm` | Weight and bias storage |
| `init_weights.asm` | Deterministic initialization |
| `dataset.asm` | Image and label buffers |
| `loader.asm` | MNIST loading and normalization |
| `logger.asm` | Console output formatting |
| `loss.asm` | Negative log loss |
| `activation.asm` | ReLU |
| `softmax.asm` | Stable softmax |
| `argmax.asm` | Predicted-class lookup |
| `dot_product.asm` | SIMD dot product kernel |
| `matrix_ops.asm` | SIMD helper kernels for outer products and matrix-vector ops |
| `reference_model.py` | Python baseline used to compare behavior and final accuracy |

## Build and run

### Docker

This is the easiest path on macOS:

```bash
cd "/path/to/mnist-asm-nn"
docker buildx build --platform linux/amd64 --load -t mnist-assembly .
docker run --rm -it --platform linux/amd64 -v "$PWD":/mnt/project -w /mnt/project mnist-assembly ./build.sh
docker run --rm -it --platform linux/amd64 -v "$PWD":/mnt/project -w /mnt/project mnist-assembly ./mnist_deep
```

### Native Linux

```bash
chmod +x build.sh
./build.sh
./mnist_deep
```

### Python reference

```bash
python3 reference_model.py
```

Both the assembly binary and the Python reference now print a final run summary at the end, including validation accuracy across epochs and final test accuracy.

## Requirements

- NASM
- GNU `ld`
- Linux x86-64 runtime
- MNIST dataset files under `dataset/`

## Notes

- The hot paths use packed SIMD operations, but the implementation is closer to SSE-style vectorization than AVX-512-specific code.
- Validation is done on the last 1,000 test samples each epoch.
- Final test accuracy is computed on all 10,000 test samples.
- If you compare against another Python implementation, match initialization, batch size, learning rate, and decay schedule before comparing metrics.
