; memory.asm — Activation buffers and gradient accumulators for 4-layer network
; Architecture: 784 → 256 → 128 → 64 → 10

section .bss

; ---- Layer activations ----
global z1, h1, z2, h2, z3, h3, o

z1 resd 256             ; pre-activation hidden layer 1
h1 resd 256             ; activation hidden layer 1 (ReLU)
z2 resd 128             ; pre-activation hidden layer 2
h2 resd 128             ; activation hidden layer 2 (ReLU)
z3 resd 64              ; pre-activation hidden layer 3
h3 resd 64              ; activation hidden layer 3 (ReLU)
o  resd 10              ; output logits (softmax applied in-place)

; ---- Gradient accumulators (accumulated over a mini-batch) ----
global dW1, dbias1, dW2, dbias2, dW3, dbias3, dW4, dbias4

dW1    resd 784*256     ; gradients for W1
dbias1 resd 256         ; gradients for b1
dW2    resd 256*128     ; gradients for W2
dbias2 resd 128         ; gradients for b2
dW3    resd 128*64      ; gradients for W3
dbias3 resd 64          ; gradients for b3
dW4    resd 64*10       ; gradients for W4
dbias4 resd 10          ; gradients for b4

; ---- Backprop intermediate gradients ----
global grad_h1, grad_h2, grad_h3, grad_o

grad_h1 resd 256        ; gradient for h1 (reused as grad_z1 after relu_backward)
grad_h2 resd 128        ; gradient for h2 (reused as grad_z2 after relu_backward)
grad_h3 resd 64         ; gradient for h3 (reused as grad_z3 after relu_backward)
grad_o  resd 10         ; gradient for output
