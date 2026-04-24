; logger.asm — Output formatting for training progress
; Provides: print_loss, print_epoch, print_accuracy, print_val_accuracy, print_summary_header

section .data
    newline db 10
    epoch_text db "Epoch ", 0
    epoch_sep  db "/25", 10, 0
    accuracy_text db "Test accuracy: ", 0
    val_acc_text db "  Val Acc: ", 0
    percent_text db "%", 10, 0
    loss_prefix  db "  Loss: ", 0
    done_text    db "Training complete.", 10, 0
    summary_text db 10, "Run summary", 10, 0

section .bss
    buffer resb 32
    epoch_buffer resb 16

section .text
global print_loss, print_epoch, print_accuracy, print_val_accuracy, print_summary_header

; ============================================================
; print_epoch(r14 = epoch number)
; Prints: "Epoch N/15\n"
; ============================================================
print_epoch:
    push rbp
    mov rbp, rsp
    push r14

    ; Print "Epoch "
    mov rax, 1
    mov rdi, 1
    mov rsi, epoch_text
    mov rdx, 6
    syscall

    ; Convert epoch number to string
    pop rax
    mov rdi, epoch_buffer
    call int_to_string

    ; Print epoch number
    mov rsi, epoch_buffer
    call string_length
    mov rdx, rax
    mov rax, 1
    mov rdi, 1
    syscall

    ; Print "/15\n"
    mov rax, 1
    mov rdi, 1
    mov rsi, epoch_sep
    mov rdx, 4
    syscall

    pop rbp
    ret

; ============================================================
; print_loss(xmm0 = loss as double)
; Prints: "  Loss: X.XXXX\n"
; ============================================================
print_loss:
    push rbp
    mov rbp, rsp

    sub rsp, 8
    movsd [rsp], xmm0

    ; Print "  Loss: "
    push rax
    mov rax, 1
    mov rdi, 1
    mov rsi, loss_prefix
    mov rdx, 8
    syscall
    pop rax

    ; Convert double to string
    mov rdi, buffer
    movsd xmm0, [rsp]
    call double_to_string

    ; Print the number
    mov rsi, buffer
    call string_length
    mov rdx, rax
    mov rax, 1
    mov rdi, 1
    syscall

    ; Print newline
    mov rax, 1
    mov rdi, 1
    mov rsi, newline
    mov rdx, 1
    syscall

    add rsp, 8
    pop rbp
    ret

; ============================================================
; print_accuracy(xmm0 = accuracy ratio as double, e.g. 0.949)
; Prints: "Test accuracy: XX.XXXX\n"
; ============================================================
print_accuracy:
    push rbp

    sub rsp, 8
    movsd [rsp], xmm0

    mov rbp, rsp

    ; Print "Test accuracy: "
    lea rsi, [rel accuracy_text]
    xor rax, rax
.count:
    cmp byte [rsi + rax], 0
    je .len_done
    inc rax
    jmp .count
.len_done:

    mov rdx, rax
    mov rsi, accuracy_text
    mov rax, 1
    mov rdi, 1
    syscall

    movsd xmm0, [rsp]
    add rsp, 8

    call print_loss_value

    pop rbp
    ret

; ============================================================
; print_val_accuracy(xmm0 = accuracy ratio as double)
; Prints: "  Val Acc: XX.XXXX%\n"
; ============================================================
print_val_accuracy:
    push rbp
    mov rbp, rsp

    sub rsp, 8
    movsd [rsp], xmm0

    ; Print "  Val Acc: "
    mov rax, 1
    mov rdi, 1
    mov rsi, val_acc_text
    mov rdx, 11
    syscall

    ; Multiply by 100 for percentage display
    movsd xmm0, [rsp]
    mulsd xmm0, [rel hundred]

    ; Convert to string
    mov rdi, buffer
    call double_to_string

    ; Print
    mov rsi, buffer
    call string_length
    mov rdx, rax
    mov rax, 1
    mov rdi, 1
    syscall

    ; Print "%\n"
    mov rax, 1
    mov rdi, 1
    mov rsi, percent_text
    mov rdx, 2
    syscall

    add rsp, 8
    pop rbp
    ret

; ============================================================
; Helper: print_loss_value — prints double in xmm0 then newline
; ============================================================
print_loss_value:
    push rbp
    mov rbp, rsp

    sub rsp, 8
    movsd [rsp], xmm0

    mov rdi, buffer
    movsd xmm0, [rsp]
    call double_to_string

    mov rsi, buffer
    call string_length
    mov rdx, rax
    mov rax, 1
    mov rdi, 1
    syscall

    ; newline
    mov rax, 1
    mov rdi, 1
    mov rsi, newline
    mov rdx, 1
    syscall

    add rsp, 8
    pop rbp
    ret

; ============================================================
; print_summary_header()
; ============================================================
print_summary_header:
    mov rax, 1
    mov rdi, 1
    mov rsi, summary_text
    mov rdx, 13
    syscall
    ret

; ============================================================
; ============================================================
; Convert double in xmm0 to string at rdi
; ============================================================
double_to_string:
    push rbp
    mov rbp, rsp
    push rbx

    ; Check if negative
    pxor xmm1, xmm1
    comisd xmm0, xmm1
    jae .not_negative
    mov byte [rdi], '-'
    inc rdi
    subsd xmm1, xmm0
    movsd xmm0, xmm1
.not_negative:

    ; Get integer part
    cvttsd2si rax, xmm0
    push rax

    ; Convert integer part to string
    call int_to_string

    ; Add decimal point
    mov byte [rdi], '.'
    inc rdi

    ; Get fractional part: (original - integer) * 10000
    cvtsi2sd xmm1, qword [rsp]
    movsd xmm2, xmm0
    subsd xmm2, xmm1
    mulsd xmm2, [scale]
    cvttsd2si rax, xmm2

    ; Convert fractional part (4 digits)
    mov rcx, 4
.frac_loop:
    mov rbx, 10
    xor rdx, rdx
    div rbx
    add dl, '0'
    mov [rdi + rcx - 1], dl
    loop .frac_loop

    add rdi, 4
    mov byte [rdi], 0

    pop rax
    pop rbx
    pop rbp
    ret

; ============================================================
; Convert integer in rax to string at rdi
; ============================================================
int_to_string:
    push rbx
    push rcx
    push rdx

    mov rbx, 10
    test rax, rax
    jnz .not_zero
    mov byte [rdi], '0'
    inc rdi
    jmp .done

.not_zero:
    mov rcx, 0
.digit_loop:
    xor rdx, rdx
    div rbx
    add dl, '0'
    push rdx
    inc rcx
    test rax, rax
    jnz .digit_loop

.pop_loop:
    pop rax
    mov [rdi], al
    inc rdi
    loop .pop_loop

.done:
    mov byte [rdi], 0
    pop rdx
    pop rcx
    pop rbx
    ret

; ============================================================
; Get length of null-terminated string in rsi → rax
; ============================================================
string_length:
    xor rax, rax
.count:
    cmp byte [rsi + rax], 0
    je .done
    inc rax
    jmp .count
.done:
    ret

section .data
scale   dq 10000.0
hundred dq 100.0
