import argparse
import os
import numpy as np


def load_images(filename):
    with open(filename, "rb") as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return (data.astype(np.float32) / 255.0).reshape(-1, 784)


def load_labels(filename):
    with open(filename, "rb") as f:
        f.read(8)
        return np.frombuffer(f.read(), dtype=np.uint8)


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes, dtype=np.float32)[np.asarray(targets).reshape(-1)]


class MLP:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        rng = np.random.default_rng(1337)
        self.W1 = rng.uniform(-0.0505, 0.0505, (784, 256)).astype(np.float32)
        self.b1 = np.zeros(256, dtype=np.float32)
        self.W2 = rng.uniform(-0.0884, 0.0884, (256, 128)).astype(np.float32)
        self.b2 = np.zeros(128, dtype=np.float32)
        self.W3 = rng.uniform(-0.1250, 0.1250, (128, 64)).astype(np.float32)
        self.b3 = np.zeros(64, dtype=np.float32)
        self.W4 = rng.uniform(-0.1768, 0.1768, (64, 10)).astype(np.float32)
        self.b4 = np.zeros(10, dtype=np.float32)

        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        self.vel = [np.zeros_like(p) for p in self.params]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-7
        self.t = 0

    def forward(self, x):
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.maximum(0, self.z2)
        self.z3 = self.h2 @ self.W3 + self.b3
        self.h3 = np.maximum(0, self.z3)
        self.o = self.h3 @ self.W4 + self.b4
        exp_o = np.exp(self.o - np.max(self.o, axis=1, keepdims=True))
        self.p = exp_o / np.sum(exp_o, axis=1, keepdims=True)
        return self.p

    def backward(self, y_true):
        grad_o = self.p - y_true
        dW4 = self.h3.T @ grad_o
        db4 = np.sum(grad_o, axis=0)
        grad_h3 = grad_o @ self.W4.T
        grad_z3 = grad_h3 * (self.z3 > 0)
        dW3 = self.h2.T @ grad_z3
        db3 = np.sum(grad_z3, axis=0)
        grad_h2 = grad_z3 @ self.W3.T
        grad_z2 = grad_h2 * (self.z2 > 0)
        dW2 = self.h1.T @ grad_z2
        db2 = np.sum(grad_z2, axis=0)
        grad_h1 = grad_z2 @ self.W2.T
        grad_z1 = grad_h1 * (self.z1 > 0)
        dW1 = self.x.T @ grad_z1
        db1 = np.sum(grad_z1, axis=0)
        return [dW1, db1, dW2, db2, dW3, db3, dW4, db4]

    def update(self, grads, lr, batch_size):
        avg_grads = [g.astype(np.float32) / np.float32(batch_size) for g in grads]
        if self.optimizer == "sgd":
            for p, g in zip(self.params, avg_grads):
                p -= np.float32(lr) * g
        elif self.optimizer == "momentum":
            beta = np.float32(0.9)
            for p, g, v in zip(self.params, avg_grads, self.vel):
                v *= beta
                v += g
                p -= np.float32(lr) * v
        else:
            self.t += 1
            b1 = np.float32(self.beta1)
            b2 = np.float32(self.beta2)
            one = np.float32(1.0)
            lr32 = np.float32(lr)
            for p, g, m, v in zip(self.params, avg_grads, self.m, self.v):
                m *= b1
                m += (one - b1) * g
                v *= b2
                v += (one - b2) * (g * g)
                m_hat = m / (one - np.float32(self.beta1 ** self.t))
                v_hat = v / (one - np.float32(self.beta2 ** self.t))
                p -= lr32 * (m_hat / (np.sqrt(v_hat) + np.float32(self.eps)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "momentum", "adam"], default="sgd")
    args = parser.parse_args()

    data_dir = "dataset"
    train_images = load_images(os.path.join(data_dir, "train-images.idx3-ubyte"))
    train_labels = load_labels(os.path.join(data_dir, "train-labels.idx1-ubyte"))
    test_images = load_images(os.path.join(data_dir, "t10k-images.idx3-ubyte"))
    test_labels = load_labels(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))

    mlp = MLP(args.optimizer)
    epochs = 25
    batch_size = 64
    lr_defaults = {"sgd": 0.005, "momentum": 0.0005, "adam": 0.0005}
    lr = lr_defaults[args.optimizer]
    decay = 0.97
    epoch_losses = []
    epoch_val_accs = []

    print("Architecture: 784 -> 256 -> 128 -> 64 -> 10")
    print(f"Optimizer: {args.optimizer.capitalize()}")
    print(f"LR: {lr}, Batch: {batch_size}, Epochs: {epochs}, Decay: {decay}\n")

    for epoch in range(epochs):
        loss_sum = 0.0
        num_batches = len(train_images) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            x_batch = train_images[start:end]
            y_batch = train_labels[start:end]
            y_batch_onehot = get_one_hot(y_batch, 10)
            p = mlp.forward(x_batch)
            batch_loss = -np.sum(y_batch_onehot * np.log(p + 1e-9)) / batch_size
            loss_sum += batch_loss
            grads = mlp.backward(y_batch_onehot)
            mlp.update(grads, lr, batch_size)

        val_x = test_images[9000:10000]
        val_y = test_labels[9000:10000]
        val_p = mlp.forward(val_x)
        val_preds = np.argmax(val_p, axis=1)
        val_acc = np.mean(val_preds == val_y)
        epoch_loss = loss_sum / num_batches
        epoch_losses.append(epoch_loss)
        epoch_val_accs.append(val_acc)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss:.4f}  Val Acc: {val_acc*100:.2f}%")
        lr *= decay

    test_p = mlp.forward(test_images)
    test_preds = np.argmax(test_p, axis=1)
    test_acc = np.mean(test_preds == test_labels)

    print("\nRun summary")
    print(f"Optimizer: {args.optimizer.capitalize()}")
    for i, (loss, val_acc) in enumerate(zip(epoch_losses, epoch_val_accs), start=1):
        print(f"Epoch {i}/{epochs}  Loss: {loss:.4f}  Val Acc: {val_acc*100:.2f}%")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
