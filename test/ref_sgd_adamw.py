# /// script
# requires-python = ">=3.10"
# dependencies = ["torch"]
# ///
"""PyTorch reference: SGD+momentum and AdamW on XOR MLP.
Uses identical init as C (seed=42), loss values must match step-for-step.

Usage:
  uv run --script test/ref_sgd_adamw.py
"""
import torch
import torch.nn as nn

# ── XOR data ──
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([0, 1, 1, 0])

# ── identical init as C's tensor_uniform(seed=42) ──
l1_w = [-0.7066419125, 0.0347714014, 0.3329391181, -0.3347364962, -0.1750457138, -0.4295166731, 0.6729872823, 0.0174204726, 0.0430614352, -0.3435101807, -0.5556625128, 0.4461668730, 0.5664554834, -0.0678417608, -0.3600747585, -0.3572188616, -0.4408469200, -0.2498578578, -0.5607960224, 0.4346924722, 0.0486644357, 0.4871314168, 0.3351954818, -0.5963274837, 0.0551676117, -0.5218325853, 0.5126883388, -0.0500682220, -0.0396361612, -0.0707036555, -0.3769234419, -0.6896869540]
l1_b = [-0.6745434999, 0.6979279518, 0.5875196457, 0.4042352140, 0.1001543999, 0.3807634711, 0.1747546494, -0.2203002721, -0.1758812368, -0.3293533325, -0.2096757591, 0.1993264407, -0.1926439703, -0.6324262619, 0.0405287445, -0.4840861261]
l2_w = [-0.0230907053, -0.0853604823, -0.1536096632, -0.2176329345, 0.2432630956, 0.0230436325, -0.2054354399, 0.2466512024, -0.0331349522, 0.1007820070, -0.1566298604, 0.0218508840, 0.2477149367, -0.1550192088, 0.0921312273, -0.0503082424, -0.0305156410, 0.1235103905, -0.1609653533, 0.1553907990, 0.1530492604, -0.2012694180, -0.2351510078, -0.1830148995, 0.0687178075, -0.0598337203, -0.1253047287, 0.0033623278, 0.0105790198, -0.1982363462, 0.2417995036, -0.0757045150]
l2_b = [0.1341633201, -0.1173271090]

def make_model():
    model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor(l1_w).reshape(2, 16).T)
        model[0].bias.copy_(torch.tensor(l1_b))
        model[2].weight.copy_(torch.tensor(l2_w).reshape(16, 2).T)
        model[2].bias.copy_(torch.tensor(l2_b))
    return model

def run(label, opt_ctor, epochs=200):
    model = make_model()
    opt = opt_ctor(model.parameters())
    print(f"── XOR MLP ({label}) ──")
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(X)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if epoch % 40 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:3d}, loss {loss.item():.6f}")
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        print(f"  accuracy: {(preds == y).sum().item()}/4\n")

run("SGD, lr=0.1, momentum=0.9",
    lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9))

run("AdamW, lr=0.01, wd=0.01",
    lambda p: torch.optim.AdamW(p, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01))
