; backward_pass.asm — Backpropagation for 4-layer network (784→256→128→64→10)
; Gradient chain: grad_o → W4 → grad_h3 → relu(z3) → W3 → grad_h2 → relu(z2) → W2 → grad_h1 → relu(z1) → W1

global accumulate_gradients, relu_backward, softmax_cross_entropy_backward
extern img_float, label
extern h1, h2, h3, o
extern z1, z2, z3
extern W1, W2, W3, W4
extern dW1, dbias1, dW2, dbias2, dW3, dbias3, dW4, dbias4
extern grad_h1, grad_h2, grad_h3, grad_o
extern outer_product_add, matrix_vector_multiply

section .data
one dd 1.0

section .text

accumulate_gradients:
    push rbp
    mov rbp, rsp

    ; ==================== BACKPROPAGATION ====================

    ; ---- Output layer gradient (softmax + cross entropy) ----
    lea rdi, [rel o]
    lea rsi, [rel label]
    lea rdx, [rel grad_o]
    mov rcx, 10
    call softmax_cross_entropy_backward

    ; ---- Layer 4 gradients (W4, b4) ----
    ; dW4 += grad_o ⊗ h3  (outer product)
    lea rdi, [rel grad_o]       ; gradient from output (size 10)
    lea rsi, [rel h3]           ; input to layer 4 (size 64)
    lea rdx, [rel dW4]
    mov rcx, 10                 ; output size (grad_o size)
    mov r9, 64                  ; input size (h3 size)
    call outer_product_add

    ; dbias4 += grad_o
    mov rcx, 10
    xor rax, rax
.accumulate_db4_loop:
    movss xmm0, [grad_o + rax*4]
    addss xmm0, [dbias4 + rax*4]
    movss [dbias4 + rax*4], xmm0
    inc rax
    cmp rax, rcx
    jl .accumulate_db4_loop

    ; grad_h3 = grad_o × W4^T (propagate gradient back through W4)
    lea rdi, [rel grad_o]       ; gradient (size 10)
    lea rsi, [rel W4]           ; weights (64×10)
    lea rdx, [rel grad_h3]     ; output gradient (size 64)
    mov rcx, 10                 ; grad_o size
    mov r9, 64                  ; W4 columns
    call matrix_vector_multiply

    ; ---- Layer 3 gradients with ReLU ----
    ; Apply ReLU backward: gate grad_h3 by z3 (pre-activation, NOT h3)
    lea rdi, [rel z3]           ; pre-activation z3
    lea rsi, [rel grad_h3]     ; gradient from above
    mov rcx, 64
    call relu_backward          ; grad_h3 now contains grad_z3

    ; dW3 += grad_z3 ⊗ h2
    lea rdi, [rel grad_h3]     ; gradient (grad_z3, size 64)
    lea rsi, [rel h2]           ; input to layer 3 (size 128)
    lea rdx, [rel dW3]
    mov rcx, 64                 ; output size
    mov r9, 128                 ; input size
    call outer_product_add

    ; dbias3 += grad_z3
    mov rcx, 64
    xor rax, rax
.accumulate_db3_loop:
    movss xmm0, [grad_h3 + rax*4]
    addss xmm0, [dbias3 + rax*4]
    movss [dbias3 + rax*4], xmm0
    inc rax
    cmp rax, rcx
    jl .accumulate_db3_loop

    ; grad_h2 = grad_z3 × W3^T
    lea rdi, [rel grad_h3]     ; gradient (grad_z3, size 64)
    lea rsi, [rel W3]           ; weights (128×64)
    lea rdx, [rel grad_h2]     ; output gradient (size 128)
    mov rcx, 64                 ; grad size
    mov r9, 128                 ; W3 columns
    call matrix_vector_multiply

    ; ---- Layer 2 gradients with ReLU ----
    lea rdi, [rel z2]           ; pre-activation z2
    lea rsi, [rel grad_h2]
    mov rcx, 128
    call relu_backward          ; grad_h2 now contains grad_z2

    ; dW2 += grad_z2 ⊗ h1
    lea rdi, [rel grad_h2]     ; gradient (grad_z2, size 128)
    lea rsi, [rel h1]           ; input to layer 2 (size 256)
    lea rdx, [rel dW2]
    mov rcx, 128                ; output size
    mov r9, 256                 ; input size
    call outer_product_add

    ; dbias2 += grad_z2
    mov rcx, 128
    xor rax, rax
.accumulate_db2_loop:
    movss xmm0, [grad_h2 + rax*4]
    addss xmm0, [dbias2 + rax*4]
    movss [dbias2 + rax*4], xmm0
    inc rax
    cmp rax, rcx
    jl .accumulate_db2_loop

    ; grad_h1 = grad_z2 × W2^T
    lea rdi, [rel grad_h2]     ; gradient (grad_z2, size 128)
    lea rsi, [rel W2]           ; weights (256×128)
    lea rdx, [rel grad_h1]     ; output gradient (size 256)
    mov rcx, 128                ; grad size
    mov r9, 256                 ; W2 columns
    call matrix_vector_multiply

    ; ---- Layer 1 gradients with ReLU ----
    lea rdi, [rel z1]           ; pre-activation z1
    lea rsi, [rel grad_h1]
    mov rcx, 256
    call relu_backward          ; grad_h1 now contains grad_z1

    ; dW1 += grad_z1 ⊗ img
    lea rdi, [rel grad_h1]     ; gradient (grad_z1, size 256)
    lea rsi, [rel img_float]   ; input image (size 784)
    lea rdx, [rel dW1]
    mov rcx, 784                ; size of img
    mov r9, 256                 ; size of grad_z1
    call outer_product_add

    ; dbias1 += grad_z1
    mov rcx, 256
    xor rax, rax
.accumulate_db1_loop:
    movss xmm0, [grad_h1 + rax*4]
    addss xmm0, [dbias1 + rax*4]
    movss [dbias1 + rax*4], xmm0
    inc rax
    cmp rax, rcx
    jl .accumulate_db1_loop

    pop rbp
    ret

; relu_backward(z, grad, size)
; rdi = pre-activation z (gates on whether z > 0)
; rsi = gradient from above (modified in-place)
; rcx = size
relu_backward:
    push rbp
    mov rbp, rsp
    xor rax, rax
.relu_backward_loop:
    movss xmm0, [rdi + rax*4]  ; z[i]
    xorps xmm1, xmm1
    comiss xmm0, xmm1
    jbe .zero_grad
    movss xmm0, [rsi + rax*4]  ; gradient from above
    jmp .store_grad
.zero_grad:
    xorps xmm0, xmm0
.store_grad:
    movss [rsi + rax*4], xmm0
    inc rax
    cmp rax, rcx
    jl .relu_backward_loop
    pop rbp
    ret

; softmax_cross_entropy_backward(probs, label, grad_out, num_classes)
; rdi = output probabilities
; rsi = true label pointer
; rdx = gradient output
; rcx = num_classes
softmax_cross_entropy_backward:
    push rbp
    mov rbp, rsp

    movzx rax, byte [rsi]      ; true label
    xor r8, r8
.softmax_grad_loop:
    movss xmm0, [rdi + r8*4]   ; p_i
    cmp r8, rax
    jne .not_true_class
    subss xmm0, [rel one]
    jmp .store_grad
.not_true_class:
.store_grad:
    movss [rdx + r8*4], xmm0
    inc r8
    cmp r8, rcx
    jl .softmax_grad_loop
    pop rbp
    ret
