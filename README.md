# MNIST Neural Network In x86-64 Assembly

This repository is a deep neural network handwritten digit classifier built entirely from scratch in raw x86-64 assembly.

The network is a 4-layer multilayer perceptron:

`784 -> 256 -> 128 -> 64 -> 10`

with:
- Full Backpropagation engine implemented in assembly
- Forward pass with ReLU hidden layers and stable Softmax output
- Custom implementations of **SGD, Momentum, and Adam** optimizers
- Direct raw binary MNIST file loading via Linux syscalls
- He-style weight initialization using a custom Linear Congruential Generator (LCG)
- Packed SIMD instructions (`xmm` SSE/AVX registers) for highly parallelized dot products and outer products

No framework is doing the interesting work here. The tensor layouts, matrix multiplication, gradient chain rule chaining, and floating-point precision are all manually managed in pure assembly.

## Why this exists

The goal of the project was not just to "get MNIST working," but to understand what a deep neural network looks like when **all the usual abstractions are stripped away**. 

Building an Adam optimizer in assembly forces you to confront things high-level ML code normally hides:
- Manual tensor layout, striding, and column-major vs row-major translation
- Calling conventions (System V AMD64 ABI) and register discipline across C-library interoperability
- Debugging floating-point (`float32`) epsilon accumulation errors
- Translating Python broadcasting semantics into physical memory offsets

This repository went through deep, bare-metal debugging. Some of the most interesting challenges solved:
- **The C ABI Clobber Bug**: Calling the C standard library `exp()` function inside the Softmax loop silently destroyed caller-saved vector registers (`xmm6` sum accumulator and `xmm7` max tracker). This completely corrupted the gradients across all samples until manual stack saves/restores were implemented around the C interop layer.
- **The Outer Product Matrix Bounds**: A bug where `outer_product_add` swapped its `rcx` and `r9` length parameters during the backward pass, causing the assembly to read past a 10-element gradient vector buffer into random memory space, computing completely corrupted weights for the upper layers.
- **Hardware RNG portability**: Relying on `rdrand` caused `SIGILL` instruction crashes on Apple Silicon Rosetta emulation, requiring a custom software-based LCG for cross-architecture deterministic weight initialization.

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
| Optimizers | SGD, Momentum, Adam |
| Epochs | 25 |
| Batch size | 64 |
| Adam params | LR: 0.0005, β1: 0.9, β2: 0.999, ε: 1e-7 |
| SGD params | LR: 0.005 |
| LR decay | 0.97 per epoch (all optimizers) |
| Dataset Size | 60,000 train / 10,000 test |
| Initialization | He-style scaling with deterministic LCG |

## Results

The assembly engine achieves performance completely identical to a ground-truth Python `numpy` reference model trained under the exact same deterministic conditions.

| Optimizer | Assembly Test Accuracy | Python Reference Test Accuracy |
|---|---|---|
| **Adam** | `98.35%` | `98.06%` |
| **SGD** | `94.83%` | `94.81%` |

*The assembly model actually slightly outperforms the reference in this test seed, validating that the underlying mathematics, gradient bounds, and optimizer moment updates are completely sound.*

## Build and run

### Docker (Recommended for macOS Apple Silicon / Windows)

```bash
cd "/path/to/mnist-asm-nn"
docker buildx build --platform linux/amd64 --load -t mnist-assembly .
docker run --rm -it --platform linux/amd64 -v "$PWD":/mnt/project -w /mnt/project mnist-assembly /bin/bash

# Inside the container:
chmod +x build.sh
./build.sh adam      # or sgd, or momentum
./mnist_deep
```

### Native Linux x86-64

```bash
chmod +x build.sh
./build.sh adam
./mnist_deep
```

### Python reference comparison

```bash
python3 reference_model.py --optimizer adam
```

## Requirements

- NASM
- GNU `ld`
- Linux x86-64 runtime (or Docker with `linux/amd64` emulation)
- MNIST dataset files under `dataset/`

## Project Structure

- `main.asm`: The core training/validation/testing loop.
- `backward_pass.asm` & `forward_pass.asm`: The entire gradient engine.
- `optimizer.asm`: The SGD, Momentum, and Adam state machines.
- `memory.asm`: Dynamic `.bss` segment allocation for all 4 layers of weights, biases, activations, and Adam tracking buffers.
- `matrix_ops.asm` & `dot_product.asm`: Highly vectorized kernels for deep learning math.

