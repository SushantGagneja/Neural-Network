; main.asm — Entry point for MNIST 4-layer neural network
; Architecture: 784 → 256 → 128 → 64 → 10
; Training: batch_size=64, epochs=25, lr=0.005 with 0.97 decay

global _start
extern load_mnist_image, load_mnist_label, fetch_data, fetch_labels
extern layer_forward, softmax, neg_log
extern img, label, img_float
extern W1, b1, W2, b2, W3, b3, W4, b4
extern z1, h1, z2, h2, z3, h3, o
extern dW1, dbias1, dW2, dbias2, dW3, dbias3, dW4, dbias4
extern grad_h1, grad_h2, grad_h3, grad_o
extern accumulate_gradients, update_weights, clear_gradients
extern print_loss, print_epoch, print_accuracy, print_val_accuracy, print_summary_header
extern argmax
extern init_weights
extern learning_rate, decay_factor

BATCH_SIZE equ 64
EPOCHS equ 25
TOTAL_SAMPLES equ (60000 / BATCH_SIZE) * BATCH_SIZE
BATCHES_PER_EPOCH equ TOTAL_SAMPLES / BATCH_SIZE
TOTAL_SAMPLES_TEST equ 10000
VAL_START equ 9000              ; validation from index 9000
VAL_COUNT equ 1000              ; validate on 1000 samples

section .bss
losses resd BATCH_SIZE          ; store per-sample losses
epoch_val_history resq EPOCHS
final_test_accuracy resq 1

section .text
_start:
    ; ---- Initialize weights with He initialization ----
    call init_weights

    ; ---- Load training data ----
    mov r15, EPOCHS
    push 0                      ; 0 = train data
    call fetch_data
    call fetch_labels
    add rsp, 8

.epoch_loop:
    push r15
    mov r14, EPOCHS + 1
    sub r14, r15
    call print_epoch
    xor r14, r14                ; batch index = 0

.batch_loop:
    push r14
    xor rbx, rbx                ; sample index = 0

    ; Calculate global sample index: (batch_index * BATCH_SIZE)
    mov rax, r14
    imul rax, BATCH_SIZE
    mov r13, rax                ; r13 = base index for this batch

.sample_loop:
    ; load image and label
    push rbx
    push r13

    mov rax, r13
    add rax, rbx                ; global sample index
    call load_mnist_image

    pop r13
    pop rbx
    push rbx
    push r13

    mov rax, r13
    add rax, rbx                ; global sample index
    call load_mnist_label

    ; ============== FORWARD PASS (4 layers) ==============

    ; Layer 1: img_float → h1 (256 neurons, ReLU)
    lea rdi, [rel img_float]
    lea rsi, [rel W1]
    lea rdx, [rel b1]
    lea r8,  [rel h1]
    mov rcx, 256
    mov r9, 784
    lea rax, [rel z1]           ; z-buffer pointer
    push rax                    ; arg 8: z_buf
    push 1                      ; arg 7: use_relu = true
    call layer_forward
    add rsp, 16

    ; Layer 2: h1 → h2 (128 neurons, ReLU)
    lea rdi, [rel h1]
    lea rsi, [rel W2]
    lea rdx, [rel b2]
    lea r8,  [rel h2]
    mov rcx, 128
    mov r9, 256
    lea rax, [rel z2]
    push rax
    push 1
    call layer_forward
    add rsp, 16

    ; Layer 3: h2 → h3 (64 neurons, ReLU)
    lea rdi, [rel h2]
    lea rsi, [rel W3]
    lea rdx, [rel b3]
    lea r8,  [rel h3]
    mov rcx, 64
    mov r9, 128
    lea rax, [rel z3]
    push rax
    push 1
    call layer_forward
    add rsp, 16

    ; Layer 4: h3 → o (10 neurons, no activation — softmax applied separately)
    lea rdi, [rel h3]
    lea rsi, [rel W4]
    lea rdx, [rel b4]
    lea r8,  [rel o]
    mov rcx, 10
    mov r9, 64
    push 0                      ; arg 8: z_buf = NULL (no z needed for output layer)
    push 0                      ; arg 7: use_relu = false
    call layer_forward
    add rsp, 16

    ; Softmax
    lea rdi, [rel o]
    lea rsi, [rel o]
    mov rcx, 10
    call softmax

    ; Loss = -log(p[label])
    movzx rdi, byte [rel label]
    movss xmm0, [o + rdi*4]
    cvtss2sd xmm0, xmm0        ; float -> double
    call neg_log
    cvtsd2ss xmm0, xmm0        ; double -> float

    pop r13
    pop rbx

    movss [losses + rbx*4], xmm0

    call accumulate_gradients   ; gradients for this sample

    ; Next sample
    inc rbx
    cmp rbx, BATCH_SIZE
    jl .sample_loop

    ; ---- End of batch ----

    ; Average loss for batch
    pxor xmm1, xmm1
    xor rbx, rbx
.sum_loop:
    addss xmm1, [losses + rbx*4]
    inc rbx
    cmp rbx, BATCH_SIZE
    jl .sum_loop

    mov rax, BATCH_SIZE
    cvtsi2ss xmm0, rax
    divss xmm1, xmm0           ; avg loss in xmm1
    cvtss2sd xmm0, xmm1        ; convert to double
    call print_loss

    ; Update weights with averaged gradients
    call update_weights
    call clear_gradients

    ; Next batch
    pop r14
    inc r14
    cmp r14, BATCHES_PER_EPOCH
    jl .batch_loop

    ; ---- End of epoch: apply LR decay ----
    movss xmm0, [rel learning_rate]
    mulss xmm0, [rel decay_factor]
    movss [rel learning_rate], xmm0

    ; ---- Per-epoch validation (last 1000 test samples) ----
    ; Save epoch counter, load test data
    push 1                      ; 1 = test data
    call fetch_data
    call fetch_labels
    add rsp, 8

    xor rbx, rbx               ; val sample index (0..999)
    xor r12, r12                ; correct counter

.val_sample_loop:
    push r12
    push rbx

    mov rax, rbx
    add rax, VAL_START          ; actual test index = 9000 + rbx
    call load_mnist_image

    pop rbx
    push rbx

    mov rax, rbx
    add rax, VAL_START
    call load_mnist_label

    ; Forward pass (4 layers, no z-buffer needed for inference)
    lea rdi, [rel img_float]
    lea rsi, [rel W1]
    lea rdx, [rel b1]
    lea r8,  [rel h1]
    mov rcx, 256
    mov r9, 784
    push 0                      ; z_buf = NULL
    push 1                      ; use_relu = true
    call layer_forward
    add rsp, 16

    lea rdi, [rel h1]
    lea rsi, [rel W2]
    lea rdx, [rel b2]
    lea r8,  [rel h2]
    mov rcx, 128
    mov r9, 256
    push 0
    push 1
    call layer_forward
    add rsp, 16

    lea rdi, [rel h2]
    lea rsi, [rel W3]
    lea rdx, [rel b3]
    lea r8,  [rel h3]
    mov rcx, 64
    mov r9, 128
    push 0
    push 1
    call layer_forward
    add rsp, 16

    lea rdi, [rel h3]
    lea rsi, [rel W4]
    lea rdx, [rel b4]
    lea r8,  [rel o]
    mov rcx, 10
    mov r9, 64
    push 0
    push 0
    call layer_forward
    add rsp, 16

    lea rdi, [rel o]
    lea rsi, [rel o]
    mov rcx, 10
    call softmax

    lea rdi, [rel o]
    mov rcx, 10
    call argmax

    pop rbx
    pop r12
    movzx rdx, byte [rel label]
    cmp rax, rdx
    jne .val_no_inc
    inc r12
.val_no_inc:

    inc rbx
    cmp rbx, VAL_COUNT
    jl .val_sample_loop

    ; Print validation accuracy
    cvtsi2ss xmm0, r12
    mov rax, VAL_COUNT
    cvtsi2ss xmm1, rax
    divss xmm0, xmm1
    cvtss2sd xmm0, xmm0
    mov rax, EPOCHS
    sub rax, [rsp]
    lea r10, [rel epoch_val_history]
    movsd [r10 + rax*8], xmm0
    call print_val_accuracy

    ; Reload training data for next epoch
    push 0
    call fetch_data
    call fetch_labels
    add rsp, 8

    ; Next epoch
    pop r15
    dec r15
    jnz .epoch_loop

    ; =========================
    ;; FINAL TEST (full 10000 samples)

    push 1
    call fetch_data
    call fetch_labels
    add rsp, 8

    xor rbx, rbx               ; sample index for test
    xor r12, r12                ; correct counter

.test_sample_loop:
    mov rsi, rbx
    push r12
    push rbx

    mov rax, rbx
    call load_mnist_image

    pop rbx
    push rbx
    mov rsi, rbx

    mov rax, rbx
    call load_mnist_label

    ; Forward pass for TEST data (4 layers)
    lea rdi, [rel img_float]
    lea rsi, [rel W1]
    lea rdx, [rel b1]
    lea r8,  [rel h1]
    mov rcx, 256
    mov r9, 784
    push 0
    push 1
    call layer_forward
    add rsp, 16

    lea rdi, [rel h1]
    lea rsi, [rel W2]
    lea rdx, [rel b2]
    lea r8,  [rel h2]
    mov rcx, 128
    mov r9, 256
    push 0
    push 1
    call layer_forward
    add rsp, 16

    lea rdi, [rel h2]
    lea rsi, [rel W3]
    lea rdx, [rel b3]
    lea r8,  [rel h3]
    mov rcx, 64
    mov r9, 128
    push 0
    push 1
    call layer_forward
    add rsp, 16

    lea rdi, [rel h3]
    lea rsi, [rel W4]
    lea rdx, [rel b4]
    lea r8,  [rel o]
    mov rcx, 10
    mov r9, 64
    push 0
    push 0
    call layer_forward
    add rsp, 16

    ; Softmax
    lea rdi, [rel o]
    lea rsi, [rel o]
    mov rcx, 10
    call softmax

    ; get predicted label (argmax of o)
    lea rdi, [rel o]
    mov rcx, 10
    call argmax

    ; compare with true label
    pop rbx
    pop r12
    movzx rdx, byte [rel label]
    cmp rax, rdx
    jne .no_increment
    inc r12
.no_increment:

    inc rbx
    cmp rbx, TOTAL_SAMPLES_TEST
    jne .test_sample_loop

    ; compute accuracy
    cvtsi2ss xmm0, r12
    mov rax, TOTAL_SAMPLES_TEST
    cvtsi2ss xmm1, rax
    divss xmm0, xmm1

    cvtss2sd xmm0, xmm0
    movsd [rel final_test_accuracy], xmm0
    call print_summary_header

    xor r13, r13
.summary_loop:
    cmp r13, EPOCHS
    jge .summary_done
    mov r14, r13
    inc r14
    call print_epoch
    lea r10, [rel epoch_val_history]
    movsd xmm0, [r10 + r13*8]
    call print_val_accuracy
    inc r13
    jmp .summary_loop

.summary_done:
    movsd xmm0, [rel final_test_accuracy]
    call print_accuracy

    ; exit
    mov rax, 60
    xor rdi, rdi
    syscall
