global dot_product
section .text
; dot_product(x, W, length, bias)
; rdi = x
; rsi = W
; rcx = length
; rdx = bias
; xmm0 = result
dot_product:
    xorps xmm0, xmm0        ; accumulator
    xor rax, rax            ; index

    cmp rcx, 4
    jb .tail

.dp_loop:
    movups xmm1, [rdi + rax*4]
    movups xmm2, [rsi + rax*4]
    mulps xmm1, xmm2
    addps xmm0, xmm1
    add rax, 4
    mov rbx, rcx
    sub rbx, rax
    cmp rbx, 4
    jae .dp_loop

    ; Horizontal add xmm0: [A, B, C, D] -> sum
    haddps xmm0, xmm0       ; [A+B, C+D, A+B, C+D]
    haddps xmm0, xmm0       ; [A+B+C+D, ...]

    jmp .tail_end

.tail:
    xorps xmm0, xmm0

.tail_end:
    xorps xmm2, xmm2        ; tail accumulator
.tail_loop:
    cmp rax, rcx
    jge .add_bias
    movss xmm3, [rdi + rax*4]
    movss xmm4, [rsi + rax*4]
    mulss xmm3, xmm4
    addss xmm2, xmm3
    inc rax
    jmp .tail_loop

.add_bias:
    addss xmm0, xmm2
    movss xmm1, [rdx]
    addss xmm0, xmm1
    ret