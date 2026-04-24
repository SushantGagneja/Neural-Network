; forward_pass.asm — Single-layer forward pass with optional ReLU and z-buffer storage
; Modified to save pre-activation values (z) for correct backpropagation

global layer_forward
extern dot_product
extern relu

section .text
; layer_forward(x, W, b, out, num_neurons, input_size, use_relu, z_buf)
; rdi = pointer to input vector
; rsi = pointer to weights matrix (flattened row-major)
; rdx = pointer to bias vector
; r8  = pointer to output buffer (h = post-activation)
; rcx = num_neurons
; r9  = input_size
; [rsp+8]  = use_relu flag (1 = use ReLU, 0 = no activation)
; [rsp+16] = pointer to z-buffer (pre-activation storage, 0 = don't store)

layer_forward:
    push r15                  ; save r15 to use for relu flag
    push r14                  ; save r14 to use for z-buffer pointer
    mov r15, [rsp+24]         ; get use_relu flag (offset: +8 for r14, +8 for r15, +8 for ret addr)
    mov r14, [rsp+32]         ; get z-buffer pointer

    xor r10, r10              ; neuron index
.layer_loop:
    ; compute offset = r10 * r9 * 4
    mov rax, r10
    imul rax, r9
    shl rax, 2
    lea r11, [rsi + rax]      ; W_row = W + offset

    ; bias pointer = b + r10*4
    mov rax, r10
    shl rax, 2
    lea r12, [rdx + rax]

    ; output pointer = out + r10*4
    lea r13, [r8 + rax]

    push rsi
    push rdx
    push rcx
    push r15                  ; save relu flag
    push r14                  ; save z-buffer pointer

    ; call dot_product
    mov rdi, rdi     ; x
    mov rsi, r11     ; W_row
    mov rcx, r9      ; input_size
    mov rdx, r12     ; bias

    call dot_product          ; result in xmm0 = z (pre-activation)

    pop r14                   ; z-buffer pointer
    pop r15                   ; relu flag

    ; Store pre-activation z if z-buffer is provided
    test r14, r14             ; check if z_buf != NULL
    jz .skip_z_store
    mov rax, r10
    shl rax, 2
    movss [r14 + rax], xmm0  ; z_buf[neuron_index] = z
.skip_z_store:

    test r15, r15             ; check if use_relu != 0
    jz .skip_relu             ; if 0, skip relu

    call relu                 ; if 1, apply relu

.skip_relu:
    movss [r13], xmm0

    pop rcx
    pop rdx
    pop rsi

    inc r10
    cmp r10, rcx
    jl .layer_loop
    pop r14
    pop r15
    ret