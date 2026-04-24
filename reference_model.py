import numpy as np
import os

def load_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.astype(np.float32) / 255.0

def load_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

data_dir = "dataset"
train_images = load_images(os.path.join(data_dir, "train-images.idx3-ubyte")).reshape(-1, 784)
train_labels = load_labels(os.path.join(data_dir, "train-labels.idx1-ubyte"))
test_images = load_images(os.path.join(data_dir, "t10k-images.idx3-ubyte")).reshape(-1, 784)
test_labels = load_labels(os.path.join(data_dir, "t10k-labels.idx1-ubyte"))

class MLP:
    def __init__(self):
        # Use uniform distribution to exactly match the simplified Assembly PRNG
        self.W1 = np.random.uniform(-0.0505, 0.0505, (784, 256))
        self.b1 = np.zeros(256)
        
        self.W2 = np.random.uniform(-0.0884, 0.0884, (256, 128))
        self.b2 = np.zeros(128)
        
        self.W3 = np.random.uniform(-0.1250, 0.1250, (128, 64))
        self.b3 = np.zeros(64)
        
        self.W4 = np.random.uniform(-0.1768, 0.1768, (64, 10))
        self.b4 = np.zeros(10)

    def forward(self, x):
        self.x = x
        self.z1 = np.dot(x, self.W1) + self.b1
        self.h1 = np.maximum(0, self.z1)
        
        self.z2 = np.dot(self.h1, self.W2) + self.b2
        self.h2 = np.maximum(0, self.z2)
        
        self.z3 = np.dot(self.h2, self.W3) + self.b3
        self.h3 = np.maximum(0, self.z3)
        
        self.o = np.dot(self.h3, self.W4) + self.b4
        
        # Softmax
        exp_o = np.exp(self.o - np.max(self.o, axis=1, keepdims=True))
        self.p = exp_o / np.sum(exp_o, axis=1, keepdims=True)
        return self.p

    def backward(self, y_true):
        batch_size = y_true.shape[0]
        
        grad_o = self.p - y_true
        
        self.dW4 = np.dot(self.h3.T, grad_o)
        self.db4 = np.sum(grad_o, axis=0)
        
        grad_h3 = np.dot(grad_o, self.W4.T)
        grad_z3 = grad_h3 * (self.z3 > 0)
        
        self.dW3 = np.dot(self.h2.T, grad_z3)
        self.db3 = np.sum(grad_z3, axis=0)
        
        grad_h2 = np.dot(grad_z3, self.W3.T)
        grad_z2 = grad_h2 * (self.z2 > 0)
        
        self.dW2 = np.dot(self.h1.T, grad_z2)
        self.db2 = np.sum(grad_z2, axis=0)
        
        grad_h1 = np.dot(grad_z2, self.W2.T)
        grad_z1 = grad_h1 * (self.z1 > 0)
        
        self.dW1 = np.dot(self.x.T, grad_z1)
        self.db1 = np.sum(grad_z1, axis=0)

    def update(self, lr, batch_size):
        # Match assembly exactly (avg grad scale)
        self.W1 -= lr * (self.dW1 / batch_size)
        self.b1 -= lr * (self.db1 / batch_size)
        
        self.W2 -= lr * (self.dW2 / batch_size)
        self.b2 -= lr * (self.db2 / batch_size)
        
        self.W3 -= lr * (self.dW3 / batch_size)
        self.b3 -= lr * (self.db3 / batch_size)
        
        self.W4 -= lr * (self.dW4 / batch_size)
        self.b4 -= lr * (self.db4 / batch_size)

mlp = MLP()
epochs = 25
batch_size = 64
lr = 0.005
decay = 0.97
epoch_losses = []
epoch_val_accs = []

print(f"Architecture: 784 -> 256 -> 128 -> 64 -> 10")
print(f"LR: {lr}, Batch: {batch_size}, Epochs: {epochs}, Decay: {decay}\n")

for epoch in range(epochs):
    loss_sum = 0
    # No shuffling to match assembly!
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
        
        mlp.backward(y_batch_onehot)
        mlp.update(lr, batch_size)
    
    lr *= decay
    
    # Validation uses last 1000 test samples
    val_x = test_images[9000:10000]
    val_y = test_labels[9000:10000]
    val_p = mlp.forward(val_x)
    val_preds = np.argmax(val_p, axis=1)
    val_acc = np.mean(val_preds == val_y)

    epoch_loss = loss_sum / num_batches
    epoch_losses.append(epoch_loss)
    epoch_val_accs.append(val_acc)
    print(f"Epoch {epoch+1}/{epochs}  Loss: {epoch_loss:.4f}  Val Acc: {val_acc*100:.2f}%")

test_p = mlp.forward(test_images)
test_preds = np.argmax(test_p, axis=1)
test_acc = np.mean(test_preds == test_labels)

print("\nRun summary")
for i, (loss, val_acc) in enumerate(zip(epoch_losses, epoch_val_accs), start=1):
    print(f"Epoch {i}/{epochs}  Loss: {loss:.4f}  Val Acc: {val_acc*100:.2f}%")
print(f"Test accuracy: {test_acc:.4f}")
