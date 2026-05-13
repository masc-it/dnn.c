# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy"]
# ///
"""PyTorch reference: MNIST CNN training, matching dnn.c output format.

Replicates mnist_model_forward_cnn architecture, AdamW config, and
the exact output format of mnist_train_impl from mnist.c.

Usage:
  uv run --script test/ref_mnist.py
"""
import struct
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ── Constants ──
MNIST_TRAIN_N = 60000
MNIST_TEST_N = 10000
MNIST_ROWS = 28
MNIST_COLS = 28
MNIST_PIXELS = 784
MNIST_CLASSES = 10

# ── IDX loader ──

def read_idx(path):
    """Load IDX file, return (dims[], raw_bytes)."""
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = list(struct.unpack(f">{ndim}I", f.read(ndim * 4)))
        raw = f.read()
    return dims, raw

def load_images(path):
    """Return float32 tensor [N, 784] normalized to [0,1]."""
    dims, raw = read_idx(path)
    N = dims[0]
    pixels = torch.frombuffer(bytearray(raw), dtype=torch.uint8).float().view(N, -1) / 255.0
    return pixels  # [N, 784]

def load_labels(path):
    """Return int64 tensor [N]."""
    dims, raw = read_idx(path)
    N = dims[0]
    labels = torch.frombuffer(raw, dtype=torch.uint8).long()
    return labels  # [N]

# ── Model ──

class MNIST_CNN(nn.Module):
    """Matches mnist_model_forward_cnn exactly."""

    def __init__(self):
        super().__init__()
        # conv1: 1→32, 3×3, pad=1, stride=1  → (32, 28, 28)
        self.conv1_w = nn.Parameter(torch.empty(32, 1, 3, 3))
        self.conv1_b = nn.Parameter(torch.zeros(32))
        # conv2: 32→64, 3×3, pad=1, stride=2  → (64, 14, 14)
        self.conv2_w = nn.Parameter(torch.empty(64, 32, 3, 3))
        self.conv2_b = nn.Parameter(torch.zeros(64))
        # conv3: 64→64, 3×3, pad=1, stride=2  → (64, 7, 7)
        self.conv3_w = nn.Parameter(torch.empty(64, 64, 3, 3))
        self.conv3_b = nn.Parameter(torch.zeros(64))
        # FC: 3136→128→10
        self.fc1_w = nn.Parameter(torch.empty(3136, 128))
        self.fc1_b = nn.Parameter(torch.empty(128))
        self.fc2_w = nn.Parameter(torch.empty(128, 10))
        self.fc2_b = nn.Parameter(torch.empty(10))

        self._reset_parameters()

    def _reset_parameters(self):
        """Match C init: kaiming uniform for conv, 1/sqrt(fan_in) uniform for FC."""
        # conv: U(-bound, bound), bound = sqrt(6 / fan_in)
        def _kaiming_bound(w):
            fan_in = w.shape[1] * w.shape[2] * w.shape[3]
            bound = math.sqrt(6.0 / fan_in)
            nn.init.uniform_(w, -bound, bound)

        _kaiming_bound(self.conv1_w)  # fan_in = 1*3*3 = 9
        _kaiming_bound(self.conv2_w)  # fan_in = 32*3*3 = 288
        _kaiming_bound(self.conv3_w)  # fan_in = 64*3*3 = 576
        nn.init.zeros_(self.conv1_b)
        nn.init.zeros_(self.conv2_b)
        nn.init.zeros_(self.conv3_b)

        # FC: U(-1/sqrt(in_features), 1/sqrt(in_features)) for both W and b
        def _fc_uniform(w, b):
            bound = 1.0 / math.sqrt(w.shape[0])
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -bound, bound)

        _fc_uniform(self.fc1_w, self.fc1_b)  # fan_in = 3136
        _fc_uniform(self.fc2_w, self.fc2_b)  # fan_in = 128

    def forward(self, x):
        """x: (N, 784) → logits (N, 10)"""
        N = x.shape[0]
        h = x.view(N, 1, 28, 28)                               # (N, 1, 28, 28)
        h = F.conv2d(h, self.conv1_w, self.conv1_b, stride=1, padding=1)
        h = F.relu(h)                                            # (N, 32, 28, 28)
        h = F.conv2d(h, self.conv2_w, self.conv2_b, stride=2, padding=1)
        h = F.relu(h)                                            # (N, 64, 14, 14)
        h = F.conv2d(h, self.conv3_w, self.conv3_b, stride=2, padding=1)
        h = F.relu(h)                                            # (N, 64, 7, 7)
        h = h.view(N, -1)                                        # (N, 3136)
        h = F.linear(h, self.fc1_w.T, self.fc1_b)               # (N, 128)
        h = F.relu(h)
        h = F.dropout(h, 0.5, training=self.training)           # dropout(0.5)
        h = F.linear(h, self.fc2_w.T, self.fc2_b)               # (N, 10)
        return h

# ── Training ──

def train_cnn(model, train_images, train_labels, epochs, batch_size, lr,
              val_n, patience, device):
    """Replicate mnist_train_cnn / mnist_train_impl output exactly.

    train_images: [N, 784] float32
    train_labels: [N] int64
    Last val_n samples held out as validation set.
    """
    N = train_images.shape[0]
    tr_n = N - val_n
    n_batches = (tr_n + batch_size - 1) // batch_size
    use_val = val_n > 0

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                            eps=1e-8, weight_decay=0.01)

    # Header line
    print(f"  N={N}  train={tr_n}  val={val_n}  batches={n_batches}  "
          f"batch={batch_size}  epochs={epochs}  lr={lr:.4f}", end="")
    if use_val:
        print(f"  patience={patience}", end="")
    print()

    # Early-stopping state
    best_val_acc = -1.0
    no_improve = 0
    best_state = None

    stopped_early = False
    for epoch in range(epochs):
        if stopped_early:
            break

        # shuffle training indices
        torch.manual_seed(epoch)  # deterministic shuffle per epoch
        perm = torch.randperm(tr_n).tolist()

        epoch_loss = 0.0
        batch_times = []

        for b in range(n_batches):
            start_t = time.perf_counter()

            start = b * batch_size
            end = min(start + batch_size, tr_n)
            bs = end - start

            # gather batch
            idx = perm[start:end]
            bx = train_images[idx].to(device)
            by = train_labels[idx].to(device)

            # forward
            logits = model(bx)
            loss = F.cross_entropy(logits, by)

            # backward
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_val = loss.item()
            epoch_loss += loss_val

            elapsed = time.perf_counter() - start_t
            batch_times.append(elapsed)

            if b > 0 and b % 100 == 0:
                print(f"    batch {b:4d}/{n_batches}  loss {loss_val:.6f}\r", end="", flush=True)

        avg_loss = epoch_loss / n_batches

        # Validation
        val_acc = None
        if use_val:
            model.eval()
            correct = 0
            with torch.no_grad():
                v_batch = min(batch_size, 256)
                for s in range(tr_n, N, v_batch):
                    end = min(s + v_batch, N)
                    bs = end - s
                    bx = train_images[s:end].to(device)
                    by = train_labels[s:end].to(device)
                    logits = model(bx)
                    preds = logits.argmax(dim=1)
                    correct += (preds == by).sum().item()
            val_acc = correct / val_n
            model.train()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    # restore best
                    model.load_state_dict(best_state)
                    stopped_early = True

        # Compute batch/sec
        avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0

        # Print epoch result
        print(f"  epoch {epoch + 1:3d}/{epochs}  loss {avg_loss:.6f}", end="")
        if val_acc is not None:
            print(f"  val_acc {val_acc:.4f}", end="")
            if stopped_early:
                print(f"  early stop (best {best_val_acc:.4f})", end="")
        print(f"  {batches_per_sec:.1f} batch/s")

    # Restore best if early stopped
    if use_val and best_state is not None:
        model.load_state_dict(best_state)

# ── Evaluation ──

@torch.no_grad()
def eval_cnn(model, images, labels, device):
    """Return accuracy, matching mnist_eval_generic batch=128 logic."""
    model.eval()
    N = images.shape[0]
    batch_size = 128
    correct = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        bx = images[start:end].to(device)
        by = labels[start:end].to(device)
        logits = model(bx)
        preds = logits.argmax(dim=1)
        correct += (preds == by).sum().item()
    return correct / N

# ── Main ──

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    base = "/Users/mascit/projects/dnn.c"
    print("\nLoading training data...")
    train_images = load_images(f"{base}/train-images-idx3-ubyte")
    train_labels = load_labels(f"{base}/train-labels-idx1-ubyte")
    print(f"  {train_images.shape[0]} images loaded.")

    print("Loading test data...")
    test_images = load_images(f"{base}/t10k-images-idx3-ubyte")
    test_labels = load_labels(f"{base}/t10k-labels-idx1-ubyte")
    print(f"  {test_images.shape[0]} images loaded.")

    print("\n═══ CNN ═══\n")

    model = MNIST_CNN()
    model.to(device)
    print("CNN model created.\n")

    print("Training CNN (AdamW, lr=0.001, batch=128, max_epochs=3, patience=3):")
    train_cnn(model, train_images, train_labels,
              epochs=3, batch_size=128, lr=0.001,
              val_n=5000, patience=3, device=device)

    print("\nEvaluating CNN:")
    train_acc = eval_cnn(model, train_images, train_labels, device)
    test_acc = eval_cnn(model, test_images, test_labels, device)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
