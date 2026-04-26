#!/bin/bash
set -e
OPTIMIZER="${1:-sgd}"

case "$OPTIMIZER" in
  sgd) OPT_FLAG=0 ;;
  momentum) OPT_FLAG=1 ;;
  adam) OPT_FLAG=2 ;;
  *)
    echo "Usage: ./build.sh [sgd|momentum|adam]"
    exit 1
    ;;
esac

mkdir -p obj

echo "Assembling..."
echo "Optimizer mode: $OPTIMIZER"

# Core files
nasm -f elf64 -g -F dwarf main.asm          -o obj/main.o
nasm -f elf64 -g -F dwarf forward_pass.asm   -o obj/forward_pass.o
nasm -f elf64 -g -F dwarf backward_pass.asm  -o obj/backward_pass.o
nasm -f elf64 -g -F dwarf -DOPTIMIZER_MODE=$OPT_FLAG optimizer.asm -o obj/optimizer.o

# Data layout
nasm -f elf64 -g -F dwarf memory.asm         -o obj/memory.o
nasm -f elf64 -g -F dwarf weights.asm        -o obj/weights.o
nasm -f elf64 -g -F dwarf dataset.asm        -o obj/dataset.o

# Utilities
nasm -f elf64 -g -F dwarf loader.asm         -o obj/loader.o
nasm -f elf64 -g -F dwarf logger.asm         -o obj/logger.o
nasm -f elf64 -g -F dwarf loss.asm           -o obj/loss.o
nasm -f elf64 -g -F dwarf activation.asm     -o obj/activation.o
nasm -f elf64 -g -F dwarf dot_product.asm    -o obj/dot_product.o
nasm -f elf64 -g -F dwarf matrix_ops.asm     -o obj/matrix_ops.o
nasm -f elf64 -g -F dwarf softmax.asm        -o obj/softmax.o
nasm -f elf64 -g -F dwarf argmax.asm         -o obj/argmax.o
nasm -f elf64 -g -F dwarf exp_double.asm     -o obj/exp_double.o
nasm -f elf64 -g -F dwarf init_weights.asm   -o obj/init_weights.o

echo "Linking..."

# Link
ld -o mnist_deep \
    obj/main.o obj/forward_pass.o obj/backward_pass.o obj/optimizer.o \
    obj/memory.o obj/weights.o obj/dataset.o \
    obj/loader.o obj/logger.o obj/loss.o obj/activation.o \
    obj/dot_product.o obj/matrix_ops.o obj/softmax.o obj/argmax.o \
    obj/exp_double.o obj/init_weights.o \
    -dynamic-linker /lib64/ld-linux-x86-64.so.2 -lc -lm

echo "Build complete: ./mnist_deep"
