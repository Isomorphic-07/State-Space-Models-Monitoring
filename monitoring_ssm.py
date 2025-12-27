import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



# Repro & helpers
def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = (logits >= 0).long()
    return (preds == y.long()).float().mean().item()
    #computes binary classification accuracy

@torch.no_grad()
def tv_witness_from_logits(logits: torch.Tensor, y: torch.Tensor, pi: float) -> float:
    """
    Empirical witness for TV((1-pi)P1, pi P0) using region A = {logit >= 0}.
      witness = (1-pi)*P1(A) - pi*P0(A)
    This value is a lower bound on the actual TV distance as we establish a measureable set A
    """
    A = (logits >= 0) #A = {x : logit(x) \geq 0}
    y0 = (y == 0) #masks
    y1 = (y == 1)

    # safety
    if y0.sum().item() == 0 or y1.sum().item() == 0:
        return float("nan")

    P1A = A[y1].float().mean().item()
    #A[y1] restricts to class 1 samples, P(logit \geq 0 | y = 1)
    P0A = A[y0].float().mean().item() #P(logit \geq 0 | y = 0)
    return (1.0 - pi) * P1A - pi * P0A

@torch.no_grad()
def expected_calibration_error(probs: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
  #we want to answer when the model says "I'm p confident", is it actually correct p% of time?
  #probs is the predicted probabilites for class 1
  # y is true labels
    y = y.float()
    ece = 0.0
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    #creates n_bins + 1 evenly spaced points between 0 and 1
    # so the idea here is probabilities are continuous in [0,1], so you can't check
    # this for every value of p. Hence we require probability bins
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1] #establish confidence interval
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
            #boolean selector, mask[j] = true if lo \leq probs[j] < hi
            #asks which examples fall into this confidence bin
        else:
            mask = (probs >= lo) & (probs <= hi)
        #mask is a boolean tensor of shape [N]
        if mask.any():
          #if no samples fall into this bin, skip it
          #prevents division by zero and meaningless statistics
            conf = probs[mask].mean() #probs[mask]: give me only probabilities whose mask value is True
            # so conf is the average predicted probability in this bin (confidence)
            acc = ((probs[mask] >= 0.5).float() == y[mask]).float().mean()
            #actual fraction correct in this bin
            # probs[mmask] >= 0.5 applies standard decision rule, predict 1 if \geq 0.5
            # then check if equal to true value of y[mask]
            # then present accuracy score based on mean
            ece += mask.float().mean().item() * abs((conf - acc).item())
            #this gives us a weighted average of how wrong the model's confidence is
            # across confidence bins
            #mask.float().mean().item() averages mask values
            # abs((conf - acc).item()) is how wrong the confidence is in this bin
    return float(ece)



class BinaryMechanismSSM(nn.Module):
    """
    Simple neural SSM with two transitions, plus input-dependent gate (Mamba-ish flavor).
    State update:
      f0 = nonlin(A0 s + B0 x)
      f1 = nonlin(A1 s + B1 x)
      f1_alpha = (1-alpha) f0 + alpha f1
      g = sigmoid(Wg x)
      s_{t+1} = g * f + (1-g) * s   (skip connection)
    """
    def __init__(self, input_dim: int, state_dim: int):
        super().__init__()
        self.A0 = nn.Linear(state_dim, state_dim, bias=False) #A_0 \in R^{d\times d}
        self.B0 = nn.Linear(input_dim, state_dim, bias=True) #B_0x_t + b_0 bias is added to push the state in new directions even when input is zero

        self.A1 = nn.Linear(state_dim, state_dim, bias=False)
        self.B1 = nn.Linear(input_dim, state_dim, bias=True)

        self.gate = nn.Linear(input_dim, state_dim, bias=True) #W_g x + b_g
        self.nonlin = nn.Tanh()

        # Stabilize-ish
        with torch.no_grad():
          #Here, A0, B0, A1, B1 are weight matrices which are trained and initiliased
          # randomly (via nn.Linear) and the weights are rescaled by 0.2
          #reduce spectral radius of A0 and A1, preventing unstable dynamics
          self.A0.weight.mul_(0.2)
          self.A1.weight.mul_(0.2)

    def step(self, s: torch.Tensor, x: torch.Tensor, z: int, alpha: float) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        f0 = self.nonlin(self.A0(s) + self.B0(x))
        f1 = self.nonlin(self.A1(s) + self.B1(x))
        f1a = (1.0 - alpha) * f0 + alpha * f1
        #establish dependence on the latent safety variable
        f = f0 if z == 0 else f1a
        return g * f + (1.0 - g) * s #next state

    @torch.no_grad()
    def rollout(self, x_seq: torch.Tensor, z: int, alpha: float, s0: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x_seq.shape #x_seq \in R^{B x T x d_in}, B is batch size (#sequences), T number of timesteps per sequence, _ input dim (ignored)
        #initialise the state
        s = torch.zeros(B, self.A0.in_features, device=x_seq.device) if s0 is None else s0
        #Creates all zero initial state of shape [B, state_dim] as A0 maps state_dim -> state_dim
        #A0.in_features is a property of nn.Linear layer that tells us how many input features that layer expects
        states = [s]
        for t in range(T):
            s = self.step(s, x_seq[:, t], z=z, alpha=alpha)
            states.append(s) #unrolling the state transitions
        return torch.stack(states, dim=1)  # [B, T+1, state_dim]



# Monitors

class LinearMonitor(nn.Module):
  #Linear probe
    def __init__(self, d_in: int):
        super().__init__()
        #d_in dimension of state H, state_dim
        # logit = w^\top h + b
        # takes hidden state and projects it onto a single direction w, outputs scalar score
        self.lin = nn.Linear(d_in, 1) #linear affine layer, output dimension of 1
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.lin(h).squeeze(-1) #applies the linear layer to h and removes the last dimension if its size is 1

class MLPMonitor(nn.Module):
  #Nonlinear Probe
    def __init__(self, d_in: int, hidden: int):
      #hidden dimension controls monitor capacity
        super().__init__()
        self.net = nn.Sequential(
            #Create pipeline of layers
            nn.Linear(d_in, hidden),
            nn.ReLU(), #ReLU(z) = max(0, z)
            nn.Linear(hidden, 1),
            #logit = w_2^\top ReLU(W_1h + b_1) + b_2
        )
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)
        #clean the dimensions, shape hygiene

def train_binary_classifier(model: nn.Module,
                            X_train: torch.Tensor, y_train: torch.Tensor,
                            X_val: torch.Tensor, y_val: torch.Tensor,
                            lr: float = 1e-3, wd: float = 1e-4,
                            steps: int = 2000, batch_size: int = 256,
                            device: str = "cpu") -> Dict[str, float]:
    model = model.to(device)
    #Decide on which probe to implement, linear vs non-linear
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    #Lr: learning rate
    #wd (regularization)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    #model.parameters() returns all trainable weights in the monitor
    #weight_decay = wd penalizes large weights to reduce overfitting
    n = X_train.shape[0]

    for _ in range(steps):
      #Steps is number of minibatch updates
        idx = torch.randint(0, n, (batch_size,), device=device)
        #creates tensor of shape batch_size, each entry is an integer {0,1, ..., n-1}
        xb, yb = X_train[idx], y_train[idx].float()
        logits = model(xb)
        #establish loss function via cross entropy
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        #backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
      #validation step
        val_logits = model(X_val)
        val_acc = accuracy_from_logits(val_logits, y_val) #Predicts class 1 if logit \geq 0
        val_probs = torch.sigmoid(val_logits) #predicted probability of class 1
        val_ece = expected_calibration_error(val_probs, y_val)

    return {"val_acc": val_acc, "val_ece": val_ece}



# Sparse autoencoder + sparsity sweep

class SparseAutoencoder(nn.Module):
    """
    Encoder: ReLU(enc(h)) -> u (sparse)
    Decoder: dec(u) -> h_hat
    Reconstructs its input from a compressed internal representation
    h -> u -> \hat{h}
    training objective \hat{h} \approx h
    constraint: only a small fraction of latent units should be active at a time
    i.e most entries of u should be zero or near-zero, each example uses only a few latent features
    enforced by sparsity inducing non-linearity (i.e ReLU) and explicity sparsity penalty (i.e L1 norm)
    """
    def __init__(self, d_in: int, d_code: int):
      #d_in: dimension of input vector h
      #d_code: dimension of latent code u
        super().__init__()
        self.enc = nn.Linear(d_in, d_code) #u = W_enc h + b_enc
        self.dec = nn.Linear(d_code, d_in) #\hat{h} = W_dec u + d_dec

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.enc(h)) #sparsity mechanism: u = max(0,  W_enc h + b_enc)
        #many entries of u are exactly zero -> sparsity emerges naturally

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self.encode(h)
        h_hat = self.dec(u)
        return h_hat, u

def train_sae(sae: SparseAutoencoder,
              H_train: torch.Tensor, H_val: torch.Tensor,
              l1: float,
              lr: float = 1e-3, wd: float = 1e-5,
              steps: int = 3000, batch_size: int = 256,
              device: str = "cpu") -> Dict[str, float]:
    sae = sae.to(device)
    H_train, H_val = H_train.to(device), H_val.to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=wd)
    n = H_train.shape[0]

    for _ in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=device)
        hb = H_train[idx]
        h_hat, u = sae(hb) #forward pass
        recon = F.mse_loss(h_hat, hb) #keep information
        sparsity = u.abs().mean() #encourages many entries of u to be zero
        loss = recon + l1 * sparsity
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        h_hat, u = sae(H_val)
        recon = F.mse_loss(h_hat, H_val).item() #measure how much info is preserved
        l1_code = u.abs().mean().item()
        frac_active = (u > 1e-6).float().mean().item() #measures fraction of latent units that are "active"

    return {"val_recon_mse": recon, "val_code_l1": l1_code, "val_frac_active": frac_active}


# Data generation

@dataclass
class Config:
    input_dim: int = 16
    state_dim: int = 64
    T: int = 50
    n_traj: int = 6000
    pi: float = 0.3
    val_frac: float = 0.3
    device: str = "cpu"

@torch.no_grad()
def generate_dataset(ssm: BinaryMechanismSSM, cfg: Config, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (H, y), where H := s_T (final state), y := Z
    """
    n0 = int(cfg.n_traj * cfg.pi)
    n1 = cfg.n_traj - n0

    x0 = torch.randn(n0, cfg.T, cfg.input_dim, device=cfg.device) #creates a Tensor defined by the shape filled with random numbers sampled from standard Gaussian
    x1 = torch.randn(n1, cfg.T, cfg.input_dim, device=cfg.device)

    s0 = ssm.rollout(x0, z=0, alpha=alpha)[:, -1, :] #select final state s_T for both mechanisms
    s1 = ssm.rollout(x1, z=1, alpha=alpha)[:, -1, :]

    H = torch.cat([s0, s1], dim=0).cpu() #stacks tensors along dim = 0 (rows): H = [s_T^{(0,1)}, \dots, s_T^{(0,n_0)}, s_T^{(1,1)}, \dots, s_T^{(1,n_1)}]
    #H.shape == (n0 + n1, state_dim), stored in CPU to free GPU memory
    y = torch.cat([torch.zeros(n0), torch.ones(n1)], dim=0).long() #: [0,\dots, 0, 1, \dots, 1]
    #y.shape == (n0 + n1, )

    perm = torch.randperm(H.shape[0]) #returns random permutation of integers from 0 to (n0 + n1)-1
    return H[perm], y[perm] #applies same perutation to rows of H and entries of y to ensure correct labelling

def split_train_val(X: torch.Tensor, y: torch.Tensor, val_frac: float):
    n = X.shape[0]
    n_val = int(n * val_frac)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]
    return X_train, y_train, X_val, y_val


# Experiments: capacity sweep & SAE sparsity sweep

def capacity_sweep(H_train, y_train, H_val, y_val, pi: float, device: str):
    """
    Train monitors with increasing capacity. Plot plateau.
    """
    rows = []

    # Linear baseline
    lin = LinearMonitor(H_train.shape[1])
    stats = train_binary_classifier(lin, H_train, y_train, H_val, y_val,
                                    lr=1e-2, wd=1e-4, steps=1500, device=device)

    with torch.no_grad():
        logits = lin(H_val.to(device)).cpu()
    tv_wit = tv_witness_from_logits(logits, y_val, pi=pi)
    rows.append({"model":"linear", "hidden":0, "acc":stats["val_acc"], "ece":stats["val_ece"], "tv_wit":tv_wit})

    # MLP capacities
    for h in [16, 32, 64, 128, 256, 512, 1024, 2048]: #hidden layer dimension
        mlp = MLPMonitor(H_train.shape[1], hidden=h)
        stats = train_binary_classifier(mlp, H_train, y_train, H_val, y_val,
                                        lr=1e-3, wd=1e-4, steps=2500, device=device)
        with torch.no_grad():
            logits = mlp(H_val.to(device)).cpu()
        tv_wit = tv_witness_from_logits(logits, y_val, pi=pi)
        rows.append({"model":"mlp", "hidden":h, "acc":stats["val_acc"], "ece":stats["val_ece"], "tv_wit":tv_wit})

    # Plot plateau
    # Here we are plotting the value accuracy against the hidden layer dimension of the binary classification model
    plt.figure()
    xs = [r["hidden"] if r["model"] == "mlp" else 0 for r in rows]
    ys = [r["acc"] for r in rows]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Monitor hidden width (0 = linear)")
    plt.ylabel("Validation accuracy")
    plt.title("Capacity sweep: accuracy plateaus at an empirical ceiling")
    plt.show()

    # Plot the identity check Acc ≈ pi + TV-witness (for each monitor's own region A)
    plt.figure()
    bounds = [pi + r["tv_wit"] for r in rows]
    plt.plot(bounds, ys, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("pi + TV-witness (from monitor decision region A)")
    plt.ylabel("Observed accuracy")
    plt.title("Sanity check: Acc(M) ≈ pi + (Q1(A)-Q0(A))")
    plt.show()

    return rows

def sae_sparsity_sweep(H_train, y_train, H_val, y_val, pi: float, device: str,
                       code_dim: int = 128):
    """
    Train SAE with increasing l1 penalty, then train a fixed monitor (e.g., MLP-128) on codes.
    Shows monitoring degradation under compression.
    """
    l1_grid = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2] #controls sparsity (compression knob)
    rows = []

    # Reference: best monitor on raw H (MLP-1024)
    ref = MLPMonitor(H_train.shape[1], hidden=1024)
    ref_stats = train_binary_classifier(ref, H_train, y_train, H_val, y_val,
                                        lr=1e-3, wd=1e-4, steps=2500, device=device)
    with torch.no_grad():
        ref_logits = ref(H_val.to(device)).cpu()
    ref_tv = tv_witness_from_logits(ref_logits, y_val, pi=pi)
    ref_acc = ref_stats["val_acc"]

    for l1 in l1_grid:
        sae = SparseAutoencoder(d_in=H_train.shape[1], d_code=code_dim)
        sae_stats = train_sae(sae, H_train, H_val, l1=l1,
                              lr=1e-3, wd=1e-5, steps=3500, device=device)

        with torch.no_grad():
          #Train on a sparse representation of H
            U_train = sae.encode(H_train.to(device)).cpu()
            U_val = sae.encode(H_val.to(device)).cpu()

        # Same monitor family on codes (MLP-1024)
        mon = MLPMonitor(U_train.shape[1], hidden=1024)
        mon_stats = train_binary_classifier(mon, U_train, y_train, U_val, y_val,
                                            lr=1e-3, wd=1e-4, steps=2500, device=device)
        with torch.no_grad():
            logits = mon(U_val.to(device)).cpu()
        tv_wit = tv_witness_from_logits(logits, y_val, pi=pi) #compute the TV distance from the codes

        rows.append({
            "l1": l1,
            "acc_U": mon_stats["val_acc"],
            "tv_wit_U": tv_wit,
            "recon_mse": sae_stats["val_recon_mse"],
            "frac_active": sae_stats["val_frac_active"],
        })

    # Plot accuracy on U vs sparsity
    plt.figure()
    xs = [r["l1"] for r in rows]
    ys = [r["acc_U"] for r in rows]
    plt.plot(xs, ys, marker="o")
    plt.axhline(ref_acc, linestyle="--")
    plt.xscale("symlog", linthresh=1e-4)  # nice view including 0
    plt.xlabel("SAE L1 penalty (sparsity strength)")
    plt.ylabel("Validation accuracy (monitor on SAE codes U)")
    plt.title("Sparsity sweep: compression can reduce monitorability")
    plt.legend(["Acc on U", "Reference acc on raw H"])
    plt.show()

    # Plot recon and activity to show you're actually compressing
    plt.figure()
    plt.plot(xs, [r["recon_mse"] for r in rows], marker="o")
    plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel("SAE L1 penalty")
    plt.ylabel("Reconstruction MSE")
    plt.title("SAE reconstruction error vs sparsity")
    plt.show()

    plt.figure()
    plt.plot(xs, [r["frac_active"] for r in rows], marker="o")
    plt.xscale("symlog", linthresh=1e-4)
    plt.xlabel("SAE L1 penalty")
    plt.ylabel("Fraction of active code entries")
    plt.title("SAE sparsity vs L1 penalty")
    plt.show()

    return {"ref_acc_H": ref_acc, "ref_tv_H": ref_tv, "rows": rows}


# Run

def run():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(device=device)
    #Cfg stores experiment hyperparameters (dims, number of trajctories, seq length etc.)
    #Passing device into config ensures dataset generation and rollouts happen on same device

    # Build SSM (untrained demo)
    ssm = BinaryMechanismSSM(cfg.input_dim, cfg.state_dim).to(cfg.device).eval()

    # Choose alpha in the "interesting" regime (not saturated)
    alpha = 0.1 #defines how "similar" the 2 mechanisms are
    H, y = generate_dataset(ssm, cfg, alpha=alpha)
    H_train, y_train, H_val, y_val = split_train_val(H, y, cfg.val_frac)

    print(f"\nData ready: alpha={alpha}, train={len(H_train)}, val={len(H_val)}, device={device}")

    # 1) Capacity sweep (shows plateau)
    cap_rows = capacity_sweep(H_train, y_train, H_val, y_val, pi=cfg.pi, device=device)
    best = max(cap_rows, key=lambda r: r["acc"])
    print("\n[Capacity sweep] best monitor:",
          f"{best['model']} hidden={best['hidden']}, acc={best['acc']:.3f}, tv_wit={best['tv_wit']:.3f}, pi+tv={cfg.pi+best['tv_wit']:.3f}")

    # 2) SAE sparsity sweep (shows compression hurts monitorability)
    sae_out = sae_sparsity_sweep(H_train, y_train, H_val, y_val, pi=cfg.pi, device=device, code_dim=128)
    print("\n[SAE sweep] reference acc on H:",
          f"{sae_out['ref_acc_H']:.3f} (tv_wit={sae_out['ref_tv_H']:.3f}, pi+tv={cfg.pi+sae_out['ref_tv_H']:.3f})")
    for r in sae_out["rows"]:
        print(f"  l1={r['l1']:<7} acc_U={r['acc_U']:.3f} tv_wit_U={r['tv_wit_U']:.3f} recon={r['recon_mse']:.4f} active={100*r['frac_active']:.1f}%")

run()