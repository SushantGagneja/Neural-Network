; optimizer.asm — Weight update and gradient management for 4-layer network
; Fixes: generic update_param, rep stosd for clearing, LR decay support

global update_weights, clear_gradients
global learning_rate, decay_factor
extern W1, b1, W2, b2, W3, b3, W4, b4
extern dW1, dbias1, dW2, dbias2, dW3, dbias3, dW4, dbias4

section .data
learning_rate  dd 0.005
batch_size_inv dd 0.015625      ; 1/64
decay_factor   dd 0.97

section .text

; ============================================================
; update_param(weights, gradients, count)
; Generic weight update subroutine
; rdi = pointer to weight array
; rsi = pointer to gradient array
; rcx = number of elements
; Updates: weight[i] -= lr * (gradient[i] / batch_size)
; ============================================================
update_param:
    push rbp
    mov rbp, rsp
    xor rax, rax
.update_loop:
    movss xmm0, [rdi + rax*4]      ; current weight
    movss xmm1, [rsi + rax*4]      ; accumulated gradient
    mulss xmm1, [rel batch_size_inv] ; average the gradient
    mulss xmm1, [rel learning_rate]  ; scale by learning rate
    subss xmm0, xmm1                ; weight -= lr * avg_gradient
    movss [rdi + rax*4], xmm0
    inc rax
    cmp rax, rcx
    jl .update_loop
    pop rbp
    ret

; ============================================================
; update_weights()
; Updates all weights and biases using averaged gradients
; ============================================================
update_weights:
    push rbp
    mov rbp, rsp

    ; W4 (64*10)
    lea rdi, [rel W4]
    lea rsi, [rel dW4]
    mov rcx, 64*10
    call update_param

    ; b4 (10)
    lea rdi, [rel b4]
    lea rsi, [rel dbias4]
    mov rcx, 10
    call update_param

    ; W3 (128*64)
    lea rdi, [rel W3]
    lea rsi, [rel dW3]
    mov rcx, 128*64
    call update_param

    ; b3 (64)
    lea rdi, [rel b3]
    lea rsi, [rel dbias3]
    mov rcx, 64
    call update_param

    ; W2 (256*128)
    lea rdi, [rel W2]
    lea rsi, [rel dW2]
    mov rcx, 256*128
    call update_param

    ; b2 (128)
    lea rdi, [rel b2]
    lea rsi, [rel dbias2]
    mov rcx, 128
    call update_param

    ; W1 (784*256)
    lea rdi, [rel W1]
    lea rsi, [rel dW1]
    mov rcx, 784*256
    call update_param

    ; b1 (256)
    lea rdi, [rel b1]
    lea rsi, [rel dbias1]
    mov rcx, 256
    call update_param

    pop rbp
    ret

; ============================================================
; clear_gradients()
; Zeros all gradient accumulators using rep stosd (fast bulk zero)
; ============================================================
clear_gradients:
    push rbp
    mov rbp, rsp
    push rdi                    ; save rdi (callee convention)

    xor eax, eax               ; value to store = 0

    ; Clear dW1 (784*256 floats)
    lea rdi, [rel dW1]
    mov ecx, 784*256
    rep stosd

    ; Clear dbias1 (256 floats)
    lea rdi, [rel dbias1]
    mov ecx, 256
    rep stosd

    ; Clear dW2 (256*128 floats)
    lea rdi, [rel dW2]
    mov ecx, 256*128
    rep stosd

    ; Clear dbias2 (128 floats)
    lea rdi, [rel dbias2]
    mov ecx, 128
    rep stosd

    ; Clear dW3 (128*64 floats)
    lea rdi, [rel dW3]
    mov ecx, 128*64
    rep stosd

    ; Clear dbias3 (64 floats)
    lea rdi, [rel dbias3]
    mov ecx, 64
    rep stosd

    ; Clear dW4 (64*10 floats)
    lea rdi, [rel dW4]
    mov ecx, 64*10
    rep stosd

    ; Clear dbias4 (10 floats)
    lea rdi, [rel dbias4]
    mov ecx, 10
    rep stosd

    pop rdi
    pop rbp
    ret
