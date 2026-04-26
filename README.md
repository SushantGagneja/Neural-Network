# Neural Network in x86-64 Assembly

A fully hand-written, from-scratch implementation of a multilayer perceptron trained on the MNIST handwritten digit dataset. Every component, from forward pass to backpropagation, is implemented directly in x86-64 NASM assembly with no ML libraries and links against libm solely for the exp function; all other functionality uses Linux syscalls directly.

## Architecture

```
Input       Hidden 1    Hidden 2    Hidden 3    Output
784    ->   256    ->   128    ->   64     ->   10
       ReLU        ReLU        ReLU        Softmax
```

Four fully connected layers. ReLU activations on the first three layers. Numerically stable Softmax with categorical cross-entropy loss on the output layer.

## Training configuration

| Parameter       | Value |
|-----------------|-------|
| Optimizers      | SGD, Momentum, Adam |
| Adam params     | LR: 0.0005, β1: 0.9, β2: 0.999, ε: 1e-7 |
| SGD params      | LR: 0.005 |
| Momentum params | LR: 0.001, β: 0.95 |
| Batch size      | 64 |
| Epochs          | 25 |
| LR Decay        | 0.97 per epoch |
| Weight init     | He (Kaiming) uniform via deterministic LCG |
| Loss            | Negative log-likelihood |
| Validation set  | MNIST test samples 9000-9999 |

## Results

The assembly model fully matches a ground-truth Python numpy reference model on this architecture.

| Optimizer | Assembly Test Accuracy | Python Reference Test Accuracy |
|---|---|---|
| **Adam** | `98.35%` | `98.06%` |
| **Momentum**| `97.56%` | ~`97.50%` |
| **SGD** | `94.83%` | `94.81%` |

## Repository structure

```text
.
├── main.asm              # Entry point: training loop, validation, final test
├── forward_pass.asm      # layer_forward: linear transform + optional ReLU
├── backward_pass.asm     # accumulate_gradients, relu_backward, softmax_ce_backward
├── activation.asm        # ReLU activation
├── softmax.asm           # Numerically stable softmax (with C ABI register preservation)
├── loss.asm              # Negative log-likelihood (neg_log)
├── optimizer.asm         # SGD, Momentum, and Adam weight updates
├── init_weights.asm      # He initialization via LCG random number generator
├── weights.asm           # Static weight and bias buffers (W1-W4, b1-b4)
├── matrix_ops.asm        # outer_product_add, matrix_vector_multiply (SIMD optimized)
├── dot_product.asm       # SIMD Dot product over float arrays
├── exp_double.asm        # Double-precision exp approximation wrapper
├── argmax.asm            # Argmax over float array
├── memory.asm            # Heap allocator / BSS buffers for activations and Adam moments
├── dataset.asm           # MNIST binary file parser (IDX format)
├── loader.asm            # load_mnist_image, load_mnist_label
├── logger.asm            # print_loss, print_epoch, print_accuracy
├── dataset/              # MNIST binary files (not included, see below)
├── reference_model.py    # NumPy reference implementation for cross-validation
├── build.sh              # NASM + ld build script
└── Dockerfile            # Reproducible build environment
```

## Requirements

**To build from source:**
- NASM 2.15 or later
- GNU ld (binutils)
- Linux x86-64 (ELF binary, uses Linux syscalls directly)

**To run the reference model:**
- Python 3.8+
- NumPy

**Or use Docker** (recommended, especially for macOS Apple Silicon / Windows):

```bash
docker buildx build --platform linux/amd64 --load -t neural-network-asm .
docker run --rm -it --platform linux/amd64 -v "$PWD":/app -w /app neural-network-asm /bin/bash
```

## Dataset

Download the MNIST dataset in IDX binary format and place the four files under `dataset/`:

```text
dataset/
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

The official source is [yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist/) or the mirror at [ossci-datasets.s3.amazonaws.com](https://ossci-datasets.s3.amazonaws.com/mnist/).

## Build and run

```bash
# Assemble and link (choose your optimizer: sgd, momentum, or adam)
./build.sh adam

# Train on MNIST (outputs loss per batch, val accuracy per epoch, final test accuracy)
./mnist_deep
```

Training prints batch loss, per-epoch validation accuracy on the last 1000 test samples, and final test accuracy over all 10,000 test samples.

## Reference model

`reference_model.py` is a NumPy implementation of the identical architecture. It supports three optimizers for comparison against the assembly implementation.

```bash
python3 reference_model.py --optimizer sgd
python3 reference_model.py --optimizer momentum
python3 reference_model.py --optimizer adam
```

Use this to verify gradient correctness: if the Python model converges and the assembly model does not, the discrepancy is in the assembly backprop or weight update.

## Implementation notes

**Calling convention:** All assembly routines follow the System V AMD64 ABI. Integer arguments in `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`. Additional arguments on the stack. `XMM0-XMM7` for floating-point arguments and return values. Caller-saved registers are preserved where required. *A critical fix during development involved manually pushing/popping `xmm6` and `xmm7` around the C `exp()` function call in Softmax to prevent silent gradient corruption.*

**Floating-point:** All activations, weights, gradients, and loss values are 32-bit single precision. We utilize 128-bit SSE packed instructions (`xmm` registers) for dot products and outer products. The exp approximation in `exp_double.asm` uses 64-bit intermediate precision to reduce rounding error.

**Memory:** Static buffers in the `.bss` section cover all weight matrices, bias vectors, intermediate activations (z-buffers and h-buffers), and Adam moment tracking accumulators (`m` and `v`). The entire MNIST dataset is loaded into RAM upfront to remove file I/O bottlenecks.

**He initialization:** Weights are drawn from `Uniform(-bound, bound)` where `bound = sqrt(6 / fan_in)`, implemented with a 32-bit linear congruential generator (LCG) seeded at startup rather than the `rdrand` instruction to ensure the binary safely executes via Rosetta emulation on Apple Silicon.

**Backpropagation:** The gradient chain correctly gates ReLU backwards using pre-activation values (`z-buffers`). The chain is:

```text
grad_o  (softmax + CE)
  -> dW4, db4, grad_h3
  -> relu_backward(z3) -> grad_z3
  -> dW3, db3, grad_h2
  -> relu_backward(z2) -> grad_z2
  -> dW2, db2, grad_h1
  -> relu_backward(z1) -> grad_z1
  -> dW1, db1
```

Gradients are accumulated across the batch in `accumulate_gradients`. The total gradient sum is correctly normalized by multiplying by `1/batch_size` before being applied in `update_weights` via the chosen optimizer (SGD, Momentum, or Adam).

## Known limitations

- No dropout or batch normalization.
- Training is strictly CPU-bound. While it utilizes SSE SIMD vectorization for hot paths, it does not currently implement AVX-512 extensions for maximum wide-vector throughput.

## License


MIT
