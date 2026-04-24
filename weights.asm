; weights.asm — Weight and bias storage for 4-layer network (784→256→128→64→10)
; All weights initialized to zero here; filled at runtime by init_weights (He initialization)

section .bss
global W1, b1, W2, b2, W3, b3, W4, b4

; Layer 1: 784 inputs → 256 neurons
W1 resd 784*256         ; 200,704 floats
b1 resd 256

; Layer 2: 256 inputs → 128 neurons
W2 resd 256*128         ; 32,768 floats
b2 resd 128

; Layer 3: 128 inputs → 64 neurons
W3 resd 128*64          ; 8,192 floats
b3 resd 64

; Layer 4 (output): 64 inputs → 10 neurons
W4 resd 64*10           ; 640 floats
b4 resd 10
