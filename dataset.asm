; dataset.asm — MNIST image and label buffers (renamed from mnist_data.asm)

global img
global label, labels
global img_float

section .bss
img   resb 784*60000   ; 60000 number of 28x28 images (reused for test data too, which is 10000)
img_float resd 784     ; 28x28 image converted to float32
label resb 1           ; single label
labels resb 60000
