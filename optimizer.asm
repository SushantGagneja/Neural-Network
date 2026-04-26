; optimizer.asm — SGD, Momentum, and Adam for the 4-layer network

%ifndef OPTIMIZER_MODE
%define OPTIMIZER_MODE 0
%endif

global update_weights, clear_gradients
global learning_rate, decay_factor, optimizer_name

extern W1, b1, W2, b2, W3, b3, W4, b4
extern dW1, dbias1, dW2, dbias2, dW3, dbias3, dW4, dbias4
extern vW1, vb1, vW2, vb2, vW3, vb3, vW4, vb4
extern mW1, mb1, mW2, mb2, mW3, mb3, mW4, mb4
extern sW1, sb1, sW2, sb2, sW3, sb3, sW4, sb4

section .data
batch_size_inv dd 0.015625
decay_factor   dd 0.97
momentum_beta  dd 0.95
adam_beta1     dd 0.9
adam_beta2     dd 0.999
adam_eps       dd 1.0e-7
one_f          dd 1.0
adam_beta1_pow dd 1.0
adam_beta2_pow dd 1.0

%if OPTIMIZER_MODE = 0
learning_rate  dd 0.005
optimizer_name db "SGD", 0
%elif OPTIMIZER_MODE = 1
learning_rate  dd 0.001
optimizer_name db "Momentum", 0
%else
learning_rate  dd 0.0005
optimizer_name db "Adam", 0
%endif

section .text

; rdi = weights
; rsi = gradients
; rdx = state1 (velocity or m)
; r8  = state2 (v for adam, 0 otherwise)
; rcx = count
update_param:
    push rbp
    mov rbp, rsp
    xor rax, rax
.update_loop:
    movss xmm0, [rsi + rax*4]
    mulss xmm0, [rel batch_size_inv]

%if OPTIMIZER_MODE = 0
    mulss xmm0, [rel learning_rate]
    movss xmm1, [rdi + rax*4]
    subss xmm1, xmm0
    movss [rdi + rax*4], xmm1
%elif OPTIMIZER_MODE = 1
    mulss xmm0, [rel learning_rate]
    movss xmm1, [rdx + rax*4]
    mulss xmm1, [rel momentum_beta]
    addss xmm1, xmm0
    movss [rdx + rax*4], xmm1
    movss xmm2, [rdi + rax*4]
    subss xmm2, xmm1
    movss [rdi + rax*4], xmm2
%else
    movss xmm1, [rdx + rax*4]
    mulss xmm1, [rel adam_beta1]
    movss xmm2, [rel one_f]
    subss xmm2, [rel adam_beta1]
    movaps xmm3, xmm0
    mulss xmm3, xmm2
    addss xmm1, xmm3
    movss [rdx + rax*4], xmm1

    movss xmm4, [r8 + rax*4]
    mulss xmm4, [rel adam_beta2]
    movss xmm5, [rel one_f]
    subss xmm5, [rel adam_beta2]
    movaps xmm6, xmm0
    mulss xmm6, xmm0
    mulss xmm6, xmm5
    addss xmm4, xmm6
    movss [r8 + rax*4], xmm4

    movaps xmm6, xmm1
    movss xmm7, [rel one_f]
    subss xmm7, [rel adam_beta1_pow]
    divss xmm6, xmm7

    movaps xmm7, xmm4
    movss xmm2, [rel one_f]
    subss xmm2, [rel adam_beta2_pow]
    divss xmm7, xmm2
    sqrtss xmm7, xmm7
    addss xmm7, [rel adam_eps]

    divss xmm6, xmm7
    mulss xmm6, [rel learning_rate]
    movss xmm2, [rdi + rax*4]
    subss xmm2, xmm6
    movss [rdi + rax*4], xmm2
%endif

    inc rax
    cmp rax, rcx
    jl .update_loop
    pop rbp
    ret

update_weights:
    push rbp
    mov rbp, rsp

%if OPTIMIZER_MODE = 2
    movss xmm0, [rel adam_beta1_pow]
    mulss xmm0, [rel adam_beta1]
    movss [rel adam_beta1_pow], xmm0
    movss xmm0, [rel adam_beta2_pow]
    mulss xmm0, [rel adam_beta2]
    movss [rel adam_beta2_pow], xmm0
%endif

    lea rdi, [rel W4]
    lea rsi, [rel dW4]
    lea rdx, [rel vW4]
    lea r8,  [rel sW4]
    mov rcx, 64*10
    call update_param

    lea rdi, [rel b4]
    lea rsi, [rel dbias4]
    lea rdx, [rel vb4]
    lea r8,  [rel sb4]
    mov rcx, 10
    call update_param

    lea rdi, [rel W3]
    lea rsi, [rel dW3]
    lea rdx, [rel vW3]
    lea r8,  [rel sW3]
    mov rcx, 128*64
    call update_param

    lea rdi, [rel b3]
    lea rsi, [rel dbias3]
    lea rdx, [rel vb3]
    lea r8,  [rel sb3]
    mov rcx, 64
    call update_param

    lea rdi, [rel W2]
    lea rsi, [rel dW2]
    lea rdx, [rel vW2]
    lea r8,  [rel sW2]
    mov rcx, 256*128
    call update_param

    lea rdi, [rel b2]
    lea rsi, [rel dbias2]
    lea rdx, [rel vb2]
    lea r8,  [rel sb2]
    mov rcx, 128
    call update_param

    lea rdi, [rel W1]
    lea rsi, [rel dW1]
    lea rdx, [rel vW1]
    lea r8,  [rel sW1]
    mov rcx, 784*256
    call update_param

    lea rdi, [rel b1]
    lea rsi, [rel dbias1]
    lea rdx, [rel vb1]
    lea r8,  [rel sb1]
    mov rcx, 256
    call update_param

    pop rbp
    ret

clear_buffer:
    push rdi
    xor eax, eax
    rep stosd
    pop rdi
    ret

clear_gradients:
    push rbp
    mov rbp, rsp

    lea rdi, [rel dW1]
    mov ecx, 784*256
    call clear_buffer
    lea rdi, [rel dbias1]
    mov ecx, 256
    call clear_buffer
    lea rdi, [rel dW2]
    mov ecx, 256*128
    call clear_buffer
    lea rdi, [rel dbias2]
    mov ecx, 128
    call clear_buffer
    lea rdi, [rel dW3]
    mov ecx, 128*64
    call clear_buffer
    lea rdi, [rel dbias3]
    mov ecx, 64
    call clear_buffer
    lea rdi, [rel dW4]
    mov ecx, 64*10
    call clear_buffer
    lea rdi, [rel dbias4]
    mov ecx, 10
    call clear_buffer

    pop rbp
    ret
