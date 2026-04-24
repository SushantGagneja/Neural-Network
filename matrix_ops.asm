global outer_product_add, matrix_vector_multiply
section .text

; outer_product_add(A, B, C, lenA, lenB)
; C += A^T x B
; rdi = A (r9), rsi = B (rcx), rdx = C (r9 x rcx)
outer_product_add:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14

    xor r8, r8              ; i index for vector A
.outer_loop:    
    cmp r8, r9
    jae .outer_end

    ; Broadcast A[i] into xmm0
    movss xmm0, [rdi + r8*4]
    shufps xmm0, xmm0, 0    ; broadcast to all 4 floats

    mov r12, r8
    imul r12, rcx
    shl r12, 2
    add r12, rdx          ; r12 = &C[i,0]

    xor r10, r10            ; j index for vector B

    cmp rcx, 4
    jb .tail_loop
    
.inner_loop:
    movups xmm1, [rsi + r10*4]        ; B[j..j+3]
    movups xmm2, [r12 + r10*4]        ; C[i, j..j+3]
    
    ; C += A[i] * B[j..j+3]
    mulps xmm1, xmm0
    addps xmm2, xmm1
    movups [r12 + r10*4], xmm2

    add r10, 4
    mov r11, rcx
    sub r11, r10
    cmp r11, 4
    jae .inner_loop

.tail_loop:
    movss xmm3, [rdi + r8*4]   ; A[i]
.tail_inner_loop:
    cmp r10, rcx
    jge .next_outer_i

    movss xmm4, [rsi + r10*4]  ; B[j]
    mulss xmm4, xmm3           ; A[i] * B[j]
    movss xmm5, [r12 + r10*4]
    addss xmm5, xmm4
    movss [r12 + r10*4], xmm5
    inc r10
    jmp .tail_inner_loop

.next_outer_i:
    inc r8
    jmp .outer_loop

.outer_end:
    pop r14
    pop r13
    pop r12
    pop rbp
    ret

; matrix_vector_multiply(x, W, y, len_x, len_y)
; y = x * W
; rdi = x (rcx), rsi = W (rcx x r9), rdx = y (r9)
; rcx = columns in x (len_x)
; r9 = columns in W (len_y)
matrix_vector_multiply:
    push rbp
    mov rbp, rsp
    push r12
    push r13
    push r14
    
    xor r8, r8              ; j index for output vector
.outer_loop:
    cmp r8, r9
    jae .outer_end

    mov r11, r9
    sub r11, r8
    cmp r11, 4
    jb .tail_loop

    xorps xmm0, xmm0        ; accumulator for 4 results
    xor r10, r10            ; i index for input vector
.inner_loop:
    cmp r10, rcx
    jae .store_inner

    movss xmm1, [rdi + r10*4]    ; x[i]
    shufps xmm1, xmm1, 0         ; broadcast

    mov r12, r10
    imul r12, r9
    shl r12, 2
    add r12, rsi            ; &W[i, 0]
    
    movups xmm2, [r12 + r8*4]    ; W[i, j..j+3]
    mulps xmm2, xmm1
    addps xmm0, xmm2

    inc r10
    jmp .inner_loop

.store_inner:
    movups [rdx + r8*4], xmm0

    add r8, 4
    jmp .outer_loop

.tail_loop:
    mov r13, r8          ; j = current column
.tail_loop_col:
    cmp r13, r9
    jge .tail_end

    xorps xmm0, xmm0     ; acc
    xor r10, r10         ; i = 0

.tail_inner:
    cmp r10, rcx
    jge .tail_store

    movss xmm1, [rdi + r10*4]
    
    mov r12, r10
    imul r12, r9
    add r12, r13
    shl r12, 2
    add r12, rsi
    movss xmm2, [r12]

    mulss xmm1, xmm2
    addss xmm0, xmm1

    inc r10
    jmp .tail_inner

.tail_store:
    movss [rdx + r13*4], xmm0
    inc r13
    jmp .tail_loop_col

.tail_end:
    jmp .outer_end

.outer_end:
    pop r14
    pop r13
    pop r12
    pop rbp
    ret