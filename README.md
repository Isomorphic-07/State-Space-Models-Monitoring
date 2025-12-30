# State-Space-Models-Monitoring

This repository contains code for experiments studying the **fundamental limits of monitoring latent mechanisms**
in neural state-space models (SSMs), with a focus on **information-theoretic constraints**, **monitor capacity**, and
**representational faithfulness under sparse compression**.

The experiments accompany the paper *Limits of Monitoring* and are designed to test when internal safety-relevant
variables are detectable **in principle**, rather than merely with better probes or larger monitors.

---

## Overview

We formalize monitoring as a **binary hypothesis testing problem** over internal model representations.
A latent binary variable \( Z \in \{0,1\} \) selects between two internal transition mechanisms of an SSM.
A monitor observes internal activations and attempts to infer \( Z \).

The key questions studied are:

- When is reliable monitoring possible regardless of monitor capacity?
- How does representational compression affect monitorability?
- Can increased monitor capacity overcome overlap in internal-state distributions?

---

## Key Results

The experiments demonstrate that:

- **Monitoring accuracy is upper bounded by total variation (TV) distance** between induced internal-state distributions.
- Increasing monitor capacity yields diminishing returns once the Bayes-optimal decision rule is approximated.
- **Sparse autoencoders (SAEs)** exhibit a non-monotonic effect:
  - Moderate sparsity can improve finite-sample monitor performance via denoising and improved conditioning.
  - High sparsity destroys safety-relevant information and reduces monitorability.
- Apparent improvements in TV-based witnesses after compression arise from **estimator-level effects**, not increased information.

---

## Repository Structure

