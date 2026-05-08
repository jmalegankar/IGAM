# A Unified Architecture for Memory and Exploration in Long-Horizon POMDPs
## Innovation-Gated Associative Memory (IGAM): A principled successor to LMU + E3B for Craftax, Montezuma, MineDojo, and POPGym

---

## TL;DR

- **Abandon LMU/HiPPO-LegT and replace it with a Gated-Delta-Rule fast-weight memory cell** (matrix state $W_t$, innovation-write $W_t = W_{t-1}(\alpha_t I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$, dynamic-query read $y_t = q_t^\top W_t$). This **strictly generalises** every empirically validated insight from the lmu_ppo work (gated/innovation write, dynamic $W_{\text{query}}$, residual anti-collapse) while gaining (i) matrix-associative recall (xLSTM-mLSTM, DeltaNet, RWKV-7), (ii) data-dependent timescale warping that LegT cannot represent, and (iii) hardware-efficient parallel-scan training under TBPTT.
- **The exploration bonus is not a separate module**: the same delta-rule innovation $\delta_t = v_t - W_{t-1} k_t$ that writes the memory **is** the lifelong intrinsic reward (a predictive-coding surprisal under the agent's own associative model), and a **factored count-min sketch over Finite Scalar Quantisation (FSQ) codes of inverse-dynamics features** gives the episodic bonus in $O(D\log K)$ per step — not $O(C^2)$ Sherman-Morrison. Spatial coverage (random-$\phi$) and semantic novelty (Craftax inventory) fall out as the same object applied to per-spatial-tile codes vs. global-state codes.
- **One mathematical claim** ties the proposal together: *predictive coding in an associative memory simultaneously specifies the memory write rule, the lifelong novelty signal, and (via discretisation of its key space) the episodic novelty signal* — eliminating the three orthogonal modules that doomed HSWVIME and producing a single end-to-end trainable objective in which gradients flow through deterministic, contraction-bounded operators that PPO can stabilise.

---

## Key Findings from the Literature Survey

### 1. Memory cells: the empirical landscape has shifted decisively away from LegT

**LMU/HiPPO-LegT is dominated.** Gu et al. (HiPPO, NeurIPS 2020) themselves show that LegT requires the window length $\theta$ to be *precisely matched* to sequence length; LegS removes this hyperparameter at the cost of $\sim 1/t$ scaling that conflicts cleanly with finite-window TBPTT chunks. More importantly, LegT has *no mechanism to compress nuisance time*: information from step 0 decays as $\exp(-t/\theta)$ regardless of relevance. Craftax-style "iron mined at step 50, used at step 5,000" is exactly the regime LegT cannot represent without retuning $\theta$ per task.

Three classes of newer cells **strictly subsume** the LMU's expressive content:

| Class | Representative | State | Time-warping | Associative recall | Parallel-scan |
|---|---|---|---|---|---|
| HiPPO-LegT (current arch) | LMU | $m \in \mathbb{R}^{p\times d}$ | Fixed window $\theta$ | None (unless dynamic readout is bolted on) | No |
| Selective SSM | Mamba-2 | Diagonal | Data-dep. $\Delta_t$ (Haar wavelet projection, Huang et al. 2025) | Weak (single-key per channel) | Yes |
| Matrix-memory linear attention | xLSTM-mLSTM (Beck 2024), DeltaNet (Yang 2024), Gated DeltaNet (Yang 2024), RWKV-7 (Peng 2025) | $W_t \in \mathbb{R}^{d_k \times d_v}$ | Data-dep. decay $\alpha_t$ | **Strong (covariance/outer-product)** | Yes |
| TTT | TTT-Linear / TTT-MLP (Sun 2024) | Network weights | Online SGD | Strong | Yes (mini-batch) |

**The decisive evidence for matrix-memory delta-rule cells:**

- On the MAD synthetic recall benchmark (Yang's NeurIPS '24 talk slides), DeltaNet scores 100/100/100 on three associative-recall variants vs. Mamba's 90/90/86 and GLA's 80/63/82.
- xLSTM/mLSTM with covariance update beats Mamba on Multi-Query Associative Recall and on the 1.3 B-parameter scale (Beck et al. 2024).
- DeltaNet is the **only** linear-time architecture to provably represent the key→value association update needed for in-context retrieval; the delta rule is exactly $W_t = W_{t-1}(I - \beta k_t k_t^\top) + \beta v_t k_t^\top$ — a Householder reflection plus rank-1 update with a closed-form innovation interpretation.
- RWKV-7 (Peng 2025) generalises this to a **diagonal+rank-one** update with vector-valued in-context learning rates and proves that the resulting class can represent all regular languages and perform $S_5$ state tracking — strictly more expressive than $\mathrm{TC}^0$ (Mamba/Transformer).
- For POPGym specifically, the recently published RLBenchNet survey (Hu et al. 2025) finds that *only* Transformer-XL, Gated Transformer-XL, and **Mamba-2** solve the most memory-intensive POPGym tasks, with Mamba-2 using 8× less memory than the transformer; LMU is not in the top tier.
- On Craftax-1B/1M, R2I (Samsami et al., ICLR '24) showed S4-style structured SSMs in DreamerV3's world model lift POPGym SOTA and dominate LSTM-RSSM for memory tasks.

**Critical observation for Jai's specific design.** The LMU "dynamic $W_{\text{query}}$" insight — $C_t = \mathrm{normalize}(W_q h_{t-1})$, $y_t = C_t^\top m_t$ — is *exactly* the linear-attention read $y_t = q_t^\top W_t$ when one identifies $h_{t-1} \mapsto$ query source and the multi-channel Legendre state $m \in \mathbb{R}^{p\times d}$ with a degenerate matrix memory whose values are Legendre-basis projections of the input. The gated write innovation $r_{\text{intr}} = \|u_{\text{actual}} - \mathrm{pred}\|_2$ is precisely the magnitude of the delta-rule innovation $\|v_t - W_{t-1}k_t\|$. **The lmu_ppo design is best understood as a degenerate special case of a matrix-memory delta-rule cell** in which (a) the keys are scalar projections rather than learned vectors, (b) the values are constrained to lie on a Legendre basis, and (c) decay is fixed by the LegT discretisation rather than data-dependent. Lifting all three constraints recovers Gated DeltaNet / RWKV-7. The intuition that drove the LMU work survives — it just gets implemented in a strictly more expressive cell.

### 2. Exploration: the four-axis Pareto frontier

Reading E3B (Henaff 2022), Henaff–Jiang–Raileanu's follow-up "Study of Global and Episodic Bonuses" (ICML 2023), DEIR (Wan 2023), NovelD (Zhang 2021), Curiosity-in-Hindsight (Jarrett 2023), BYOL-Explore (Guo 2022), NGU/Agent57 (Badia 2020), RIDE (Raileanu 2020), and ETD (Jiang ICLR 2025) together yields four orthogonal design axes:

1. **Episodic vs. lifelong.** Henaff et al. ICML '23 prove empirically that the *combined* (episodic × lifelong) bonus (E3B-elliptical × RND) sets MiniHack SOTA and that *neither alone* is sufficient on contextual MDPs — the episodic bonus alone fails on singleton MDPs (Montezuma) because once an episode revisits a state the bonus dies, the lifelong bonus alone fails on procedural MDPs (Craftax) because every episode has new pixels. **Both signals are mathematically necessary.**
2. **Spatial coverage vs. task-aligned discrimination.** Jai's empirical finding that random-CNN features beat task-aligned $y_t$ on MiniGrid corridors recovers Burda et al.'s 2018 result and is explained by Henaff '23: in spatial bottlenecks, count-equivalent features that uniformly distinguish all states dominate. *But* random-$\phi$ is semantically blind in Craftax (crafting a sword leaves pixels 99% identical), so the right mechanism is not random-$\phi$ but **inverse-dynamics features** (Pathak ICM, E3B-IDM) which by construction capture only the controllable subspace and are robust to both noisy-TV *and* irrelevant pixel constancy.
3. **Quadratic vs. linear scaling.** Sherman-Morrison E3B is $O(C^2)$ per step per env. With $C\!=\!512$ and 128 envs at >50 k steps/s on Craftax-JAX, this is wall-clock-prohibitive. The two scalable alternatives are: (a) NGU/Agent57 kNN over a fixed-size episodic buffer, $O(M\,C)$ per step but $M\sim 1000$, still expensive; (b) **count-based novelty over a discrete bottleneck**, $O(D)$ where $D$ is the codebook dimensionality, completely independent of $C$.
4. **Stochasticity robustness (noisy-TV).** Jarrett et al.'s Curiosity-in-Hindsight (ICML 2023) gives the principled fix: condition predictions on a *hindsight representation* of the unpredictable component, so intrinsic reward only reflects the predictable part. The simpler, equally effective practical fix used by ICM, RIDE, NGU, E3B, and DEIR is to learn features by inverse dynamics: $\phi$ is trained so that $a_t$ is recoverable from $(\phi(o_t),\phi(o_{t+1}))$. Anything in $o$ that does not affect $a$ — including noisy TVs — is by construction in $\phi$'s null space.

The **strongest published numbers** to beat:
- ObstructedMaze-2Dlhb: E3B-IDM reaches near-1.0 success at ~25 M steps; DEIR (Wan IJCAI 2023) and ETD (ICLR 2025) match or slightly exceed this.
- Craftax-classic 1 M: Improving Transformer World Models (Cohen et al., ICML 2025) — **69.66% reward / 31.77% score**, the first agent above the 65.0% human expert. DreamerV3 = 53.2%, ∆-IRIS = 35.0%, IRIS = 25.0%, vanilla PPO-RNN ≪ 50%.
- Craftax-full: SOTA reward ≈ 27.91% (Cohen 2025), prior was 5.44%; deeper achievements (gnomish mines floors) remain mostly unsolved.
- Crafter (Hafner): EMERALD (ICML 2025) = 58.1%, first model-based agent over human within 10 M steps.
- POPGym: R2I (S5-based DreamerV3) is the published memory SOTA; Mamba-2 dominates RLBenchNet's hard-memory subset.
- Montezuma: Go-Explore (1st return then explore, Nature 2021) ≫ Agent57 ≫ NGU; among non-archival methods, BYOL-Hindsight (Jarrett 2023) holds SOTA on the *sticky-actions* (stochastic) Montezuma variant.

### 3. Representation/bottleneck

The HSWVIME post-mortem identifies the failure mode precisely: SCVAE → continuous μ → MSE reconstruction → high-variance pixels dominate, semantic features wash out. The literature has converged on the diagnosis:

- **Continuous MSE bottlenecks suffer posterior collapse and dimension washout** (Van den Oord 2017 VQ-VAE motivation; HSWVIME confirmed empirically).
- **VQ-VAE codebook collapse is a real engineering tax** (commitment loss, codebook reseeding, EMA updates, dead-code resampling). For an exploration application, this is exactly the kind of fragility we cannot afford.
- **Finite Scalar Quantisation (FSQ, Mentzer ICLR '24)** is the now-preferred discrete bottleneck: project to $d \in [3,8]$ dims, bound each dim, round to $L_i \in [3,8]$ levels, implicit codebook of size $\prod L_i$, **no commitment loss, no codebook collapse by construction**. This is the right primitive.
- **Inverse-dynamics-trained $\phi$** (Pathak ICM, E3B-IDM) is non-collapsing because the inverse loss requires $\phi$ to keep enough information to recover $a$; trivial collapse fails the loss. Combining FSQ + inverse-dynamics gives a discrete, semantic-by-construction, controllable-feature bottleneck.
- **DreamerV3's categorical latent (32×32 categoricals) and IRIS/∆-IRIS/EMERALD's transformer-tokeniser pipeline** both work in Craftax precisely because the latent is discrete; the recent SOTA paper (Cohen 2025) further attributes its gains to a *nearest-neighbour tokeniser* (NNT) that keeps codewords static after creation — a stability property FSQ achieves trivially.

---

## The Proposal: Innovation-Gated Associative Memory (IGAM)

### A single mathematical object

Let $\phi: \mathcal{O} \to \mathbb{R}^{d_\phi}$ be an encoder trained by inverse dynamics, $\mathrm{FSQ}: \mathbb{R}^{d_\phi}\to \prod_{i=1}^D \{1,\dots,L_i\}$ a stop-gradient discretisation, and $W_t \in \mathbb{R}^{d_k \times d_v}$ a matrix memory state. Define keys/values/queries from a small projection of $\phi(o_t)$ and the policy's hidden state $h_t$:

$$k_t = W_K \phi(o_t),\quad v_t = W_V \phi(o_t),\quad q_t = W_Q h_t$$

The **innovation** is

$$\boxed{\;\delta_t = v_t - W_{t-1} k_t / \|k_t\|^2 \;}$$

The gated delta-rule write (Yang 2024 Gated DeltaNet, Peng 2025 RWKV-7 specialisation):

$$\boxed{\;W_t = \alpha_t \,W_{t-1}\bigl(I - \beta_t \tfrac{k_t k_t^\top}{\|k_t\|^2}\bigr) + \beta_t\, v_t k_t^\top\;}$$

with data-dependent decay $\alpha_t = \sigma(W_\alpha h_t) \in (0,1)$ (gating, *RWKV-7-style vector-valued*) and learning rate $\beta_t = \sigma(W_\beta h_t) \in (0,1)$. The read is the dynamic-$W_{\text{query}}$ generalisation:

$$y_t = q_t^\top W_t / (q_t^\top \mathbf{1}_t + \epsilon)$$

(normaliser $\mathbf{1}_t = \alpha_t \mathbf{1}_{t-1} + \beta_t k_t$, after Beck 2024 mLSTM stabilisation).

The **central claim** is that three quantities derive from this single object:

1. **Memory read** for the policy/critic: $y_t$ above. Routed to actor/critic in concatenation with $h_t$, exactly as in Jai's lmu_ppo `cat([h, m.mean(dim=1)])` direct-access pattern.
2. **Lifelong intrinsic reward** is the squared innovation magnitude:
$$r^{\text{life}}_t = \|\delta_t\|_2^2$$
This is a *predictive-coding* signal under the agent's own associative model: high if the memory has not yet learned to map $k_t \mapsto v_t$, low if it has. It is the direct generalisation of Jai's existing $r_{\text{intr}} = \|u_{\text{actual}} - \mathrm{pred}\|_2$ (lmu_ppo gated write) from a scalar Legendre channel to a full key-value matrix memory. **Crucially, $\delta_t$ is bounded** because $W_{t-1}$ has spectral radius $\le 1$ under the gated-delta contraction (Yang 2024 proves this), so $r^{\text{life}}$ is well-conditioned for PPO advantage estimation.
3. **Episodic intrinsic reward** is a factored count-min sketch over the FSQ code of $\phi(o_t)$:
$$r^{\text{epi}}_t = \frac{1}{\sqrt{\hat{N}_e(c_t)}},\quad \hat{N}_e(c_t) = \min_{i=1\dots H} \mathrm{CMS}_i[\,h_i(c_t)\,]$$
where $c_t = \mathrm{FSQ}(\phi(o_t))$, with $D\!\sim\!6$ FSQ dims of $L_i\!=\!5$ levels giving an implicit codebook of $\sim\!15{,}625$ cells (matched to MiniGrid) or $D\!=\!8$, $L_i\!=\!8$ giving $\sim\!16$M cells (matched to Craftax/Montezuma). Per-step cost is $O(DH)$, completely independent of $d_v$.

The combined bonus follows the NGU multiplicative structure (Badia 2020), which is the form Henaff 2023 ICML found dominant on MiniHack:
$$r^{\text{int}}_t = r^{\text{epi}}_t \cdot \min(\max(r^{\text{life}}_t,\,1),\,L)$$

**Why this is one object, not three:** all three quantities are defined entirely by $(\phi, k_t, v_t, q_t, W_t, \mathrm{FSQ})$. The same encoder produces the memory's input *and* the exploration features. The same delta rule that writes the memory *is* the lifelong surprisal. The same code that defines the value *is* the episodic count key. There is a single set of representation parameters, a single prediction error, and one architectural diagram.

### Memory cell decision: drop LMU entirely, go matrix-memory delta-rule

**Recommendation: replace `models/lmu.py` and `models/wyner_lmu.py` with `models/igam_cell.py` implementing Gated DeltaNet with RWKV-7-style vector-valued in-context learning rate.**

State shape, per env, per layer:
- $W_t \in \mathbb{R}^{d_k \times d_v}$ — head-factored as $H$ heads of $W_t^{(h)} \in \mathbb{R}^{d_k/H \times d_v/H}$.
- Normaliser $n_t \in \mathbb{R}^{d_k}$ for stable read.
- Optional fast-recurrent $h_t \in \mathbb{R}^{d_h}$ small RNN core for query generation (RWKV-7 keeps a small token-shift channel alongside the matrix memory; this is the analogue of the lmu_ppo policy GRU/MLP downstream of the readout).

Recommended sizes for Phase A/B (POPGym + Craftax 1M / ObstructedMaze):
- $d_k = d_v = 64$, $H = 4$, $L_{\text{layers}} = 2$ → $\sim 8\,\mathrm{KB}$ memory per layer per env. Compare to lmu_ppo $p\!=\!16$ channels × $d\!=\!64$ Legendre = same footprint.
- For Phase C (Montezuma image, MineDojo): $d_k=d_v=128$, $H=8$, $L_{\text{layers}}=4$.

**Why this preserves every lmu_ppo insight:**

- **Dynamic $W_{\text{query}}$**: The read $y_t = q_t^\top W_t$ with $q_t = W_Q h_{t-1}$ is exactly the dynamic readout, generalised from Legendre-lag selection to associative key→value retrieval. The orthogonal gain=0.01 init for $W_Q$ remains identical.
- **Gated/innovation write**: $W_t = W_{t-1} + \beta_t \delta_t k_t^\top$ with $\delta_t = v_t - W_{t-1}k_t$ is the delta-rule write; this *is* the innovation form Jai already arrived at, but in matrix-memory space.
- **`residual_scale` anti-collapse**: implemented as a small additive $\beta_{\min}>0$ floor on the data-dependent $\beta_t$ gate. The Cayley-map orthogonal $W_{\text{pre}}$ is no longer needed because the delta rule's contraction ($I - \beta k k^\top/\|k\|^2$ has spectral radius $\le 1$) provides the bounded-state guarantee for free (Yang 2024 proof).
- **TBPTT chunk-based training**: keep the existing `LMURolloutBuffer` chunked layout; replace the recurrence with the chunkwise parallel form of Yang 2024 (WY-representation Householder products), which is *strictly more efficient* than the LMU sequential scan because it parallelises over chunk length.

**Why this is paper-worthy memory contribution.** The narrative is: *we identified that the LegT prior is incompatible with variable-horizon RL and that the dynamic $W_{\text{query}}$ insight is in fact the linear-attention query, generalising LMU into the matrix-memory delta-rule family. We then show that the innovation $\delta_t$ — already used as $r_{\text{intr}}$ in our prior work — is the exact predictive-coding surprisal under the cell's own associative model, and that this lifts gated-delta architectures from language modelling into a principled exploration-aware RL memory.*

This is an honest first-principles bridge between the SSM/linear-attention literature (which is currently RL-blind, with R2I and DRAMA the only exceptions) and the intrinsic-motivation literature (which is currently architecture-agnostic and almost always uses an LSTM core).

### Exploration mechanism: factored counts on inverse-dynamics FSQ codes

**Encoder.** $\phi: o_t \to z_t \in \mathbb{R}^{d_\phi}$, CNN for image obs (Montezuma / Craftax pixel) or MLP for symbolic (Craftax-symbolic / POPGym).

**Inverse-dynamics auxiliary** (E3B-IDM / ICM / NGU):
$$\mathcal{L}_{\text{idm}} = \mathbb{E}_t\bigl[-\log p_\theta(a_t \mid \phi(o_t), \phi(o_{t+1}))\bigr]$$
This is the only loss applied to $\phi$ itself. **Crucially, $\phi$ is detached from the policy gradient** — this is the hard-won HSWVIME lesson re-applied. The inverse loss runs in a separate optimiser at lower LR, and its features are stop-gradient'd before entering the memory cell. This decouples representation learning from PPO advantage estimation, killing the gradient-fight failure.

**FSQ bottleneck.** $\phi(o)$ is projected to $d_{\text{fsq}}$ dims, bounded ($\tanh$), rounded to $L$ levels per dim. For pixel observations we apply FSQ *per spatial location* of the CNN output (a $7\times7$ grid in MiniGrid → 49 codes per frame), giving spatial coverage automatically. For symbolic Craftax observations we apply FSQ to the global $\phi$, giving semantic novelty automatically. **The same mechanism produces both behaviours by changing only the location of the FSQ layer in the encoder.** This is the unification Jai asked for.

**Episodic count.** Per-env count-min sketch with $H=4$ hashes and width $W=2^{16}$:
- Per step: `for d in range(D): cms[d, hash(c[d])] += 1`. $O(D)$ updates.
- Bonus query: `min over d of cms[d, hash(c[d])]`. $O(D)$ reads.
- Reset to 0 at episode boundary.
- Total per-step cost: $\sim 50$ ops, vs E3B's $C^2 \sim 250\text{,}000$ ops at $C\!=\!512$.

**Lifelong bonus.** $r^{\text{life}}_t = \|\delta_t\|_2^2$ from the cell's innovation. **Important correctness check:** because $W_{t-1}$ contracts ($\rho \le 1$ under gated delta), $\|\delta_t\|$ is uniformly bounded by $\|v_t\| + \|k_t\|\,\|W_{t-1}\|_{\mathrm{op}}\le 2C_\phi$; we apply Welford-style running-mean/std normalisation per-env (matching RND/E3B normalisation) before mixing.

**Combination.** $r^{\text{int}}_t = r^{\text{epi}}_t \cdot r^{\text{life}}_t$ with $\beta_{\text{int}}=0.01$, then $r_t = r^{\text{ext}}_t + \beta_{\text{int}} r^{\text{int}}_t$. This matches NGU's structural choice and Henaff '23's empirically dominant combination.

**Noisy-TV robustness** is provided by the inverse-dynamics encoder by construction (information not relevant to $a_t$ is in $\phi$'s null space and thus does not affect FSQ codes or $\delta_t$). For environments where this is insufficient (sticky-action Montezuma, MineDojo), the Curiosity-in-Hindsight upgrade (Jarrett 2023) — concatenating a hindsight representation $\psi(o_{t+1})$ to the prediction — is a one-line addition to the cell's value $v_t$.

### Representation: discrete + frozen + inverse-dynamics — three legs of the stool

**Decision tree:**
1. **MiniGrid / ObstructedMaze (small-pixel partial obs):** $\phi$ = small CNN trained by inverse dynamics, FSQ per spatial location.
2. **Craftax-symbolic (default Craftax-1M):** $\phi$ = MLP on the 9×11 tile-symbol obs + 48 state features, FSQ globally. *Inventory and skill state are the semantic axes the FSQ codebook must carve.*
3. **Craftax-pixel / Montezuma:** $\phi$ = CNN, IDM, FSQ per spatial location for spatial coverage *and* a global head over the inventory/score region for semantic coverage. Two FSQ heads, summed.
4. **MineDojo / MineCLIP:** $\phi_{\text{vision}}$ = **frozen MineCLIP visual encoder** (Fan 2022), $\phi_{\text{lang}}$ = frozen MineCLIP text encoder for goal conditioning, *only* FSQ + IDM head trained. This avoids the catastrophic representation drift of training visual features at MineDojo scale and inherits MineCLIP's video-text grounding.
5. **POPGym (low-dim discrete obs):** $\phi$ = identity or small MLP; the test is purely on the memory cell, so exploration is disabled.

**Why this avoids HSWVIME's gradient fight:**
- Encoder loss = inverse dynamics only, separate optimiser, stop-gradient before memory.
- Memory cell loss = PPO actor/critic + auxiliary innovation MSE (a tiny $\lambda_\delta \|\delta_t\|^2$ regulariser at $\lambda_\delta=0.001$, identical role to lmu_ppo's gated-write loss).
- No generative reconstruction loss anywhere — there is no decoder that has to reproduce pixels. The Wyner KL is replaced by the deterministic, contraction-bounded innovation magnitude.
- FSQ is non-trainable in the codebook sense; there is no codebook collapse mode.
- The only gradient touching the memory state $W_t$ is PPO + tiny innovation regularisation. PPO does not fight a generative sequence model because there is no generative sequence model — there is only a deterministic associative memory whose update is the innovation that PPO is *also* using as a reward.

### A clean ablation pattern (the paper's experimental table)

For the joint paper, here is the matrix the evidence is *expected* to fall on:

| Variant | POPGym (RepeatPrev) | MiniGrid-Memory-S13 | ObstrMaze-2Dlhb | Craftax-1M reward | Montezuma sticky |
|---|---|---|---|---|---|
| **IGAM (full)** | **>0.95 IQM** | **>0.95** | **>0.95 @ 10M** | **target ≥70%** | **target ≥10k** |
| − inverse-dyn φ (random CNN) | unchanged | 0.95 | 0.5 | <40% | <2k |
| − FSQ (continuous + kNN epi) | 0.9 | 0.85 | 0.6 | 50% | <5k |
| − episodic bonus (lifelong only) | 0.9 | 0.4 | 0.2 | 35% | <5k |
| − lifelong bonus (epi only) | 0.95 | 0.95 | 0.9 | 55% | <3k |
| − Gated DeltaNet (use LMU) | 0.6 | 0.95 | 0.95 | 45% | – |
| − data-dep gating ($\alpha_t,\beta_t$ const) | 0.7 | 0.9 | 0.85 | 50% | – |
| − dynamic $W_Q$ (fixed readout) | 0.5 | 0.7 | 0.7 | 40% | – |
| DreamerV3 (matched compute) | 0.7 | 0.9 | – | 53.2% | – |
| E3B-IDM (LSTM) | n/a | 0.9 | 0.95 @ 25M | <40% | <5k |
| TWM (Cohen 2025 SOTA) | n/a | n/a | n/a | 69.7% | – |

Numbers are *expected* under the hypothesis the architecture is correct; flagged appropriately.

The story this table tells is **the central paper claim**: removing any one of {inverse-dyn $\phi$, FSQ, episodic, lifelong, matrix-memory cell, data-dep gating, dynamic query} costs at least one regime. *No competitor reproduces the full set of properties.* That is what makes this paper-worthy rather than incremental.

### Anticipated failure modes, with mitigations

| Failure mode | Origin | IGAM mitigation |
|---|---|---|
| HSWVIME generative-RL gradient fight | End-to-end Wyner+PPO | No generative reconstruction loss; encoder uses IDM only with detached features into memory; memory loss is PPO + tiny innovation reg. |
| LMU LegT time-warping bottleneck | Fixed window $\theta$ | Data-dependent $\alpha_t, \beta_t$ from RWKV-7/Gated-DeltaNet — provably memory can be "frozen" by setting $\alpha_t\to 1, \beta_t\to 0$ (Mamba's input-selectivity result, Huang 2025, transfers to delta rule). |
| E3B task-aligned-discrimination trap | $y_t$-features encode task value | Episodic bonus uses $\phi$ (inverse-dyn, pre-memory), not $y_t$. Memory is for the policy; exploration is on the orthogonal IDM features. |
| Random-$\phi$ semantic blindness | Pixel-identical semantic transitions | IDM training forces $\phi$ to capture controllable state; FSQ codes change exactly when controllable state changes. Inventory updates change the FSQ code by construction. |
| Sherman-Morrison numerical instability | $O(C^2)$ rank-1 inverse maintenance | Eliminated. Episodic mechanism is count-min sketch, $O(D)$ updates, no matrix inversion. |
| VQ codebook collapse | Standard VQ-VAE | FSQ has no learnable codebook — collapse-free by construction (Mentzer 2023). |
| Posterior collapse (HSWVIME) | KL-driven prior collapse | No KL term; deterministic innovation. |
| Reward hacking the intrinsic | Agent finds a stochastic obs that maxes lifelong | IDM features mask the stochastic obs from $\phi$; lifelong bonus capped to $L=10$ in ratio mix; episodic bonus *decreases* as the cell visits the same FSQ code, providing built-in diminishing returns within an episode. |
| Numerical blow-up of $W_t$ | Repeated rank-1 updates | Gated-delta contraction $\rho(W) \le 1$; layer-norm on $y_t$ before policy/critic head (mLSTM stabilisation). |

### Phased experimental ladder

**Phase A — Memory cell validation, no exploration (4–6 weeks).**
Targets: POPGym (RepeatPrevious, Concentration, Battleship, Autoencode), MiniGrid-Memory-S13 with `MemoryStartWrapper`, BSuite memory_length / discounting_chain.
Success criteria: Match or beat R2I-S5 IQM on POPGym, ≥0.95 on MiniGrid-Memory-S13, ≥0.9 on BSuite memory tasks. Memory ablation table: IGAM vs. LMU (current), vs. Mamba-2, vs. mLSTM, vs. DeltaNet (no gating), vs. Gated DeltaNet, vs. RWKV-7. *Prediction*: IGAM ≈ Gated DeltaNet ≈ RWKV-7 > Mamba-2 > mLSTM > LMU on contextual recall; LMU competitive only on fixed-horizon recall.
*Decision threshold*: if IGAM does not beat current LMU + dynamic $W_Q$ on at least 4/6 POPGym tasks, **do not proceed**; investigate whether the Legendre-init prior is helping more than expected (in which case, hybrid the value projection $W_V$ with Legendre-basis init).

**Phase B — Exploration validation on memory + sparse reward (8–12 weeks).**
Targets: MiniGrid-DoorKey, KeyCorridor, ObstructedMaze-2Dlh / 2Dlhb (with the IDM module enabled), Craftax-1M (symbolic).
Success criteria: ObstructedMaze-2Dlhb ≥0.9 by 10M steps (E3B-IDM published is ~0.9 by 25M), Craftax-1M reward ≥60% (DreamerV3 53.2%, target near Cohen 2025's 69.66% MFRL+TWM SOTA without a transformer world model).
Ablation columns required: − IDM (random $\phi$), − FSQ (continuous + kNN epi), − episodic only, − lifelong only, − naive E3B with $y_t$ (reproduce Jai's failure), Curiosity-in-Hindsight head on/off.
*Decision threshold*: if IGAM fails to clear 50% Craftax-1M reward, the encoder/IDM auxiliary is not capturing the inventory-state subspace; switch to symbolic FSQ on inventory features alone before re-enabling pixel coverage.

**Phase C — Scale and stochasticity (12–24 weeks).**
Targets: Craftax-full 1B, Montezuma's Revenge with sticky actions (Curiosity-in-Hindsight head ON), MineDojo programmatic tasks (MineCLIP frozen encoder, IDM still active on top, FSQ on the IDM features).
Success criteria: Craftax-full reward ≥10% (current SOTA 27.91% is with MBRL+TWM and an unfair compute advantage; the model-free target is to clear PPO-RNN ≪ 5% by a wide margin and approach mid-teens %); Montezuma sticky-action ≥10k average return matching BYOL-Hindsight; MineDojo "harvest milk" / "shear sheep" ≥0.5 success matching MineCLIP-baseline.

### Honest comparison to the best alternatives Jai must consider

**Why not just DreamerV3 + RND?** DreamerV3 + ensemble disagreement (DreamerV3-XP, Coluding 2025) is the natural baseline. *It loses on three counts*: (a) DreamerV3's GRU is empirically dominated by S4 on memory tasks (R2I), and Mamba-2/Gated DeltaNet dominate S4 on contextual recall, so the memory backbone is suboptimal; (b) RND/disagreement is purely lifelong — Henaff '23 ICML proves episodic is *necessary* for contextual MDPs, which Craftax procedural-gen is; (c) DreamerV3's pixel-reconstruction loss is exactly the HSWVIME failure mode that motivates this entire research arc — it spends representation capacity on visually salient but semantically irrelevant pixels, hurting Craftax's inventory-driven exploration. EMERALD's spatial-MaskGIT decoder partially fixes (c) but exacerbates (a).

**Why not just DeltaNet + kNN episodic?** DeltaNet alone has no exploration mechanism — it is a language-model architecture. NGU-style kNN episodic on raw $\phi$ features (a) scales as $O(MC)$ with episode buffer size $M\sim 1000$, slower than CMS, (b) requires careful $\phi$ choice — and that choice is exactly the IGAM design. So "DeltaNet + kNN episodic + IDM features" is *almost* IGAM, but loses the unification: the lifelong signal and the memory write rule are decoupled, the paper has no single mathematical claim, and the reported numbers will be similar but the contribution looks like engineering. The IGAM framing — innovation $\delta_t$ as both write and surprisal — is what raises this to ICML/NeurIPS-grade *methodological* novelty.

**Why not TTT + inverse-dynamics?** TTT (Sun 2024) is intriguing because it is the most expressive of the new cells (its hidden state is a full neural network updated by SGD). But (a) the test-time SGD is unstable under PPO chunked updates — every PPO minibatch sees a slightly different test-time-trained network, so the policy gradient sees moving targets, exactly the gradient-fight pathology HSWVIME suffered with QASampler; (b) TTT's wall-clock cost per step is much higher than gated-delta and dominates training time on Craftax-JAX; (c) TTT has no clean innovation interpretation — its update is a gradient step on a self-sup loss whose magnitude is not the natural surprisal. **Recommendation**: TTT is a Phase D research direction (24+ months) once IGAM is published; it is not the right Phase A-C cell.

**What does IGAM do that none of these alternatives do?** It reduces three traditionally orthogonal modules — memory cell, lifelong novelty, episodic novelty — to a single $(W_t, \delta_t, \mathrm{FSQ}\circ\phi)$ object. Every alternative I have surveyed glues two or three modules together with bespoke loss balancing; IGAM derives all three from one predictive-coding objective. That is the paper.

---

## Recommendations

### Staged, concrete next steps

**Weeks 0–2 — Cell prototype.** Implement `models/igam_cell.py` as Gated DeltaNet (Yang 2024 reference impl in [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) is the right starting point; chunk-wise WY-form for training, recurrent for inference). Mirror the lmu_ppo `LMURolloutBuffer` layout, replacing the LMU recurrence with the chunk-parallel scan. Add the innovation $\delta_t$ as a side output. *Threshold to advance:* parity with LMU on MiniGrid-Memory-S13 with `MemoryStartWrapper` enabled.

**Weeks 2–6 — Phase A.** Run POPGym + MiniGrid-Memory + BSuite ablation suite. *Threshold to advance:* IGAM ≥ R2I numbers on ≥4 POPGym tasks; if not, debug the dynamic-$W_Q$ initialisation (try Legendre-basis $W_V$ init) before declaring the cell wrong.

**Weeks 6–14 — Phase B.** Add the IDM auxiliary, FSQ bottleneck, count-min sketch, and combined bonus. Run ObstructedMaze and Craftax-1M. *Threshold to advance:* Craftax-1M reward ≥60%; ObstructedMaze-2Dlhb ≥0.9 by 10M steps. *If* Craftax-1M < 50%, refactor: try Cohen 2025's nearest-neighbour tokeniser pipeline alongside FSQ; investigate whether the symbolic-obs FSQ is partitioning inventory or terrain.

**Weeks 14–24 — Phase C.** Add Curiosity-in-Hindsight head for stochastic environments; integrate MineCLIP frozen encoder for MineDojo. Scale to Craftax-full 1B and Montezuma sticky.

**Weeks 24+ — Paper synthesis.** Two viable framings:
1. *Single ICML/NeurIPS submission* — "IGAM: Innovation-Gated Associative Memory unifies memory and exploration for partially-observed RL." 9 pages, focus on the mathematical unification claim plus Phase A+B headline results, defer Phase C to appendix.
2. *Two-paper sequence* — (i) memory-only paper (NeurIPS): IGAM cell vs. LMU/Mamba-2/RWKV-7 on POPGym + MiniGrid-Memory; (ii) exploration paper (ICML): the innovation/FSQ/CMS unification on Craftax + Montezuma, citing (i). The two-paper sequence is safer (each has cleaner ablations), the single paper is higher-impact if Phase C numbers come in.

### Threshold-based pivots

- **If Phase A IGAM fails to beat LMU on contextual recall:** the dynamic-readout init or normalisation is wrong; do not abandon the cell, fix the head. If after 4 weeks no progress, hybrid: use IGAM matrix memory but initialise $W_V$ projection with HiPPO-LegS basis (the matrix memory then explicitly stores Legendre projections of values, recovering LMU as a strict special case). This is the principled fallback hybrid.
- **If Phase B episodic bonus fails on Craftax:** the FSQ bottleneck is partitioning the wrong subspace. Diagnostic: dump FSQ codes per state, check that crafting-a-sword changes the code. If not, train FSQ on the inverse-dynamics features only (not on $\phi$ directly) and add an action-prediction reconstruction head.
- **If Phase B lifelong bonus fails (always near zero):** the matrix memory is over-fitting to the empirical key→value mapping inside an episode (write is too aggressive). Lower $\beta_t$ floor, increase $\alpha_t$ decay, or switch from gated-delta to plain delta.
- **If Phase C Montezuma sticky fails:** add Curiosity-in-Hindsight head; if still failing, the bottleneck is that lifelong novelty alone cannot solve room-by-room exploration. Add a Go-Explore-style archive of high-novelty states, conditioned on FSQ codes (cell representation = FSQ code), recovering the Latent Go-Explore (Gallouédec 2023) variant. This is a clean architectural addition because the FSQ codes were already the cell representation we needed.

---

## Caveats

- **The expected ablation numbers in the table above are predictions, not measurements.** The Cohen 2025 TWM numbers (Craftax-classic 69.66% reward) come from a transformer world model with VQ tokens and Dyna-with-warmup; reaching them with a model-free architecture is uncertain and would itself be a contribution. A more conservative target is "match DreamerV3 (53.2%) without a generative world model, via memory + exploration alone."
- **The "single mathematical object" framing is a story choice, not a theorem.** Strictly speaking, the inverse-dynamics loss is a separate objective from PPO and the FSQ projection has no continuous gradient. The unification is *architectural* (one diagram, one set of representations) and *interpretive* (innovation = surprisal = write magnitude); a fully unified information-theoretic derivation (e.g. variational free energy across all three) is *not* claimed and would be a dishonest pitch.
- **Curiosity-in-Hindsight requires care.** Jarrett 2023's gridworld and sticky-Montezuma results are strong, but on environments with truly noisy-TV-like distractors the hindsight representation can collapse to a copy of the next observation, defeating the purpose. The mitigation (information bottleneck on the hindsight head) is well-known but adds a hyperparameter.
- **Gated DeltaNet length generalisation has known limits.** Yang 2024 reports DeltaNet's length generalisation lags GLA/RetNet/Mamba; this matters for episodes longer than the longest training chunk. RWKV-7's stronger generalisation (it solves $S_5$ with 2 layers) is the safer Phase C choice if Craftax-full episodes regularly exceed chunk length.
- **MineDojo is a known compute trap.** Even with frozen MineCLIP, training a competent agent at MineDojo programmatic-task level routinely takes >100 GPU-days. "Unlimited compute" should be reality-checked here; budget Phase C MineDojo as a stretch goal not a deliverable.
- **The episodic-counts-on-FSQ trick is not new in isolation.** Tang et al. (#Exploration, 2017) used hash-based pseudocounts; Machado et al. (2020) used SimHash; Park et al. (CQM, NeurIPS 2023) used VQ-VAE on observations for goal-space exploration. The novelty in IGAM is **the binding** of the FSQ codebook to the same encoder/feature space that produces the matrix memory's keys, *and* the use of the matrix-memory innovation as the lifelong companion. Reviewers will probe this; the right framing is "the unification, not the components."
- **Replacing LMU is a real cost.** Six months of debugging LMU stability (orthogonal init, Cayley updates, ZOH discretisation, residual_scale anti-collapse) walks out the door. Some of those bug-bashes will recur in the new cell. The compensating gain is that gated-delta has been engineered at language-model scale by multiple groups (DeltaNet, Gated DeltaNet, RWKV-7, Songlin Yang's flash-linear-attention library), and the public reference implementations are well-tested.
- **PPO + matrix memory under TBPTT is empirically under-explored.** The published Gated-DeltaNet and RWKV-7 results are language-modelling, not on-policy RL; AGaLiTe (Pramanik 2023) and DRAMA (Mamba in Dreamer) are the closest precedents, both successful but on different tasks. Plan Phase A specifically to validate this combination — if PPO advantage estimation is unstable through the matrix memory's parallel scan, the contingency is to fall back to the recurrent (single-step) form during the rollout buffer's TBPTT chunks, accepting a wall-clock cost.
