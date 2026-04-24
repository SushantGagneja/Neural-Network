global init_weights
extern W1, W2, W3, W4

section .data
    scale_w1  dd  0.0505          ; sqrt(2/784)
    scale_w2  dd  0.0884          ; sqrt(2/256)
    scale_w3  dd  0.1250          ; sqrt(2/128)
    scale_w4  dd  0.1768          ; sqrt(2/64)
    two_inv   dd  4.6566129e-10   ; 1 / (2^31)
    
    ; LCG variables
    lcg_seed  dd  1337            ; Starting seed

section .text
init_weights:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13

    lea rdi, [rel W1]
    mov rcx, 784*256
    movss xmm5, [rel scale_w1]
    call fill_weights_he

    lea rdi, [rel W2]
    mov rcx, 256*128
    movss xmm5, [rel scale_w2]
    call fill_weights_he

    lea rdi, [rel W3]
    mov rcx, 128*64
    movss xmm5, [rel scale_w3]
    call fill_weights_he

    lea rdi, [rel W4]
    mov rcx, 64*10
    movss xmm5, [rel scale_w4]
    call fill_weights_he

    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; fill_weights_he(rdi=pointer, rcx=count, xmm5=scale)
fill_weights_he:
    push rbp
    mov rbp, rsp
    xor rax, rax            ; index = 0

.fill_loop:
    cmp rax, rcx
    jge .fill_done

    ; LCG: seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
    mov ebx, [rel lcg_seed]
    mov edx, 1103515245
    imul ebx, edx
    add ebx, 12345
    and ebx, 0x7FFFFFFF
    mov [rel lcg_seed], ebx
    
    ; Shift to make it [-2^30, 2^30-1] so we have signed values
    sub ebx, 0x3FFFFFFF

    ; Convert to float
    cvtsi2ss xmm0, ebx     ; convert int32 to float
    mulss xmm0, [rel two_inv]  ; scale to roughly [-0.5, 0.5]
    
    ; Multiply by 2.0 to get to [-1.0, 1.0]
    addss xmm0, xmm0
    
    ; Scale by He factor
    mulss xmm0, xmm5       ; xmm0 = random * scale

    movss [rdi + rax*4], xmm0
    inc rax
    jmp .fill_loop

.fill_done:
    pop rbp
    ret
