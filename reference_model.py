#!/usr/bin/env python3

import numpy as np
import struct
import os

def load_images(path):
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows * cols).astype(np.float32) / 255.0

def load_labels(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

#Activation fns 
def relu(x):
    return np.maximum(0, x)

def relu_backward(z, grad):
    """Gate gradient by pre-activation z (correct way)"""
    return grad * (z > 0).astype(np.float32)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

#He initialization
def he_init(fan_in, fan_out):
    scale = np.sqrt(2.0 / fan_in)
    return np.random.uniform(-scale, scale, (fan_out, fan_in)).astype(np.float32)

#Network
class Network:
    def __init__(self):
        self.W1 = he_init(784, 256)
        self.b1 = np.zeros(256, dtype=np.float32)
        self.W2 = he_init(256, 128)
        self.b2 = np.zeros(128, dtype=np.float32)
        self.W3 = he_init(128, 64)
        self.b3 = np.zeros(64, dtype=np.float32)
        self.W4 = he_init(64, 10)
        self.b4 = np.zeros(10, dtype=np.float32)

    def forward(self, x):
        self.x = x
        self.z1 = self.W1 @ x + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.W2 @ self.h1 + self.b2
        self.h2 = relu(self.z2)
        self.z3 = self.W3 @ self.h2 + self.b3
        self.h3 = relu(self.z3)
        self.z4 = self.W4 @ self.h3 + self.b4
        self.o = softmax(self.z4)
        return self.o

    def backward(self, label):
        #output grad
        grad_o = self.o.copy()
        grad_o[label] -= 1.0

        #L4
        dW4 = np.outer(grad_o, self.h3)
        db4 = grad_o.copy()
        grad_h3 = self.W4.T @ grad_o

        #L3
        grad_z3 = relu_backward(self.z3, grad_h3)
        dW3 = np.outer(grad_z3, self.h2)
        db3 = grad_z3.copy()
        grad_h2 = self.W3.T @ grad_z3

        # L2
        grad_z2 = relu_backward(self.z2, grad_h2)
        dW2 = np.outer(grad_z2, self.h1)
        db2 = grad_z2.copy()
        grad_h1 = self.W2.T @ grad_z2

        #L1
        grad_z1 = relu_backward(self.z1, grad_h1)
        dW1 = np.outer(grad_z1, self.x)
        db1 = grad_z1.copy()

        return dW1, db1, dW2, db2, dW3, db3, dW4, db4

def main():
    #Paths
    data_dir = "dataset"
    train_images = load_images(os.path.join(data_dir, "train-images.idx3-ubyte"))
    train_labels = load_labels(os.path.join(data_dir, "train-labels.idx1-ubyte"))
    test_images = load_images(os.path.join(data_dir, "t10k-images.idx3-ubyte"))
    test_labels = load_labels(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))

    net = Network()
    lr = 0.005
    batch_size = 64
    epochs = 25

    print(f"Architecture: 784 → 256 → 128 → 64 → 10")
    print(f"LR: {lr}, Batch: {batch_size}, Epochs: {epochs}, Decay: 0.97")
    print()

    #Forward pass for debugging
    o = net.forward(train_images[0])
    print(f"Sample 0 forward pass:")
    print(f"  h1[0:5] = {net.h1[:5]}")
    print(f"  h2[0:5] = {net.h2[:5]}")
    print(f"  h3[0:5] = {net.h3[:5]}")
    print(f"  output  = {o}")
    print(f"  label   = {train_labels[0]}")
    print()

    #Training
    total_samples = (60000 // batch_size) * batch_size
    batches = total_samples // batch_size

    for epoch in range(epochs):
        epoch_loss = 0
        batch_count = 0

        for b in range(batches):
            #Accumulate grads
            dW1_acc = np.zeros_like(net.W1)
            db1_acc = np.zeros_like(net.b1)
            dW2_acc = np.zeros_like(net.W2)
            db2_acc = np.zeros_like(net.b2)
            dW3_acc = np.zeros_like(net.W3)
            db3_acc = np.zeros_like(net.b3)
            dW4_acc = np.zeros_like(net.W4)
            db4_acc = np.zeros_like(net.b4)
            batch_loss = 0

            for s in range(batch_size):
                idx = b * batch_size + s
                o = net.forward(train_images[idx])
                loss = -np.log(o[train_labels[idx]] + 1e-10)
                batch_loss += loss

                dW1, db1, dW2, db2, dW3, db3, dW4, db4 = net.backward(train_labels[idx])
                dW1_acc += dW1; db1_acc += db1
                dW2_acc += dW2; db2_acc += db2
                dW3_acc += dW3; db3_acc += db3
                dW4_acc += dW4; db4_acc += db4

            #Update wts
            net.W1 -= lr * dW1_acc / batch_size
            net.b1 -= lr * db1_acc / batch_size
            net.W2 -= lr * dW2_acc / batch_size
            net.b2 -= lr * db2_acc / batch_size
            net.W3 -= lr * dW3_acc / batch_size
            net.b3 -= lr * db3_acc / batch_size
            net.W4 -= lr * dW4_acc / batch_size
            net.b4 -= lr * db4_acc / batch_size

            epoch_loss += batch_loss / batch_size
            batch_count += 1

        # LR decay (gentler curve)
        lr *= 0.97

        #Validation 
        correct = 0
        for i in range(9000, 10000):
            o = net.forward(test_images[i])
            if np.argmax(o) == test_labels[i]:
                correct += 1

        print(f"Epoch {epoch+1}/25  Loss: {epoch_loss/batch_count:.4f}  Val Acc: {correct/10:.2f}%  LR: {lr:.6f}")

    #Final test accuracy
    correct = 0
    for i in range(10000):
        o = net.forward(test_images[i])
        if np.argmax(o) == test_labels[i]:
            correct += 1
    print(f"\nTest accuracy: {correct/100:.2f}%")

if __name__ == "__main__":
    main()
