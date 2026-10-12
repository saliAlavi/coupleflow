# Theory (draft for Sec. 3 + Appendix A)

Notation. A trial gives a neural recording x (C channels × T samples at 64 Hz) and K candidate
stimulus envelopes a_1..a_K, one of which (index y) is attended. A decoder outputs a score
s_k = S(x, a_k) per candidate and decides ŷ = argmax_k s_k, with ties broken against the model.

## Proposition 1 (null calibration of coupling-only scoring)
**Assumptions.**
(a) The score uses the neural data only through a function g shared by all candidates:
s_k = g(x, a_k).
(b) Under the null H0, x is independent of (y, a_1, …, a_K).
(c) Given y, the candidate set is exchangeable: (a_y, a_{−y}) has the same distribution as any
permutation of it.

**Claim.** Under H0, P(ŷ = y) ≤ 1/K. Equality holds when ties have probability zero.

**Proof.** By (b) and (c), the joint law of (x, a_{π(1)}, …, a_{π(K)}) does not depend on the
permutation π. So the vector (s_1, …, s_K) = (g(x, a_1), …, g(x, a_K)) is exchangeable, and
each index is the unique maximiser with the same probability, (1 − P(tie))/K. Ties are
counted as errors. ∎

*How the evaluation instantiates each assumption.*
- (a) holds by construction: every decision is "coupling-only". The side and fused paths are
  reported separately, and their Shapley credit is measured.
- (b) is realised by the across-trial EEG-shuffle null (x replaced by another trial's
  recording) and the circular-shift null (x time-shifted, which keeps its autocorrelation but
  breaks time-locking).
- (c) holds exactly for the same-talker set: the negative is the attended talker's own
  speech from a disjoint window. For the competing set it holds approximately in
  counterbalanced designs. The matched variant equalises each window's value distribution
  across candidates by quantile matching.

The observed null accuracies (49.6–50.4%, all datasets and windows) are the empirical check.

*Time-centred score.*
S(x, a) = ⟨ē, ā⟩ / (‖ē‖‖ā‖), where ē and ā are the neural and acoustic embeddings with their
own time means removed. If x is constant in time, ē = 0 and every candidate scores 0, so the
tie rule gives chance. The score therefore carries no information unless the neural signal
co-varies with the candidate over time. This is the property that rules out
"recognise-the-stimulus" shortcuts.

## Proposition 2 (the PMI decision rule is Bayes-optimal)
**Assumptions.**
(a) Given the candidates, the recording depends on the attended stream only:
p(x | a_1..a_K, y) = p(x | a_y).
(b) The prior over y is uniform.

**Claim.** The maximum-a-posteriori decision is
ŷ = argmax_k PMI(a_k; x), where PMI(a; x) = log p(a | x) − log p(a).

**Proof.**
P(y = k | x, a_{1:K}) ∝ p(x | a_k) · P(y = k) ∝ p(x | a_k) = p(a_k | x) p(x) / p(a_k).
The factor p(x) is common to all k. Taking logs gives
log P(y = k | ·) = PMI(a_k; x) + const. ∎

**Corollary (prior bias).**
- Scoring by conditional likelihood log p(a_k | x) alone adds log p(a_k), which favours
  acoustically typical candidates regardless of the brain. This is the "audio-only" shortcut
  that MERIT's probe detects.
- PMI cancels that term exactly.
- CoupleFlow estimates both terms with one conditional flow-matching network trained with
  conditioning dropout (p_uncond = 0.2). log p(a) is its unconditional mode, and both terms
  share Monte-Carlo draws, which reduces the variance of their difference.

**Per-source residuals.** For attended and unattended audio, R_c(a) is the conditional
flow-matching residual and R_u(a) the unconditional one. R_u − R_c is the flow's estimate of
PMI, so the difference [R_u − R_c](attended) − [R_u − R_c](unattended) is an attention index
that should be > 0. It is reported per dataset.

## Proposition 3 (evidence-accumulation law)
**Assumptions.**
- In a decision window of T seconds, the decoder's time course r̂ has correlations ρ_a and
  ρ_u with the attended and competing envelopes.
- Samples decorrelate at an effective rate f (effective samples per second).
- The standard large-sample approximation of Fisher's z applies:
  z(r) = atanh(r) ~ N(atanh(ρ), 1/(fT − 3)).
- The two correlations are computed on nearly uncorrelated envelopes, so their covariance is
  negligible.

**Claim.**
P(correct | T) = Φ( Δζ · √((fT − 3)/2) ), with Δζ = atanh ρ_a − atanh ρ_u.
For fT ≫ 3 this is Φ(κ √T), with κ = Δζ √(f/2).

**Proof.** The decision is correct iff z_a − z_u > 0. The difference is Gaussian with mean
Δζ and variance 2/(fT − 3), so P(z_a − z_u > 0) = Φ(Δζ / √(2/(fT − 3))). ∎

**Corollaries.**
1. **The window needed for a target accuracy p** is T_p = (Φ⁻¹(p)/κ)². Since T_0.95 ∝ κ⁻²,
   halving the window needed means doubling κ².
2. **SNR dependence.** Model the decoder output as signal plus independent noise at
   per-sample signal-to-noise ratio σ. Then ρ_a = ρ* √(σ/(1+σ)), where ρ* is the noiseless
   correlation. For small ρ, Δζ ≈ ρ_a − ρ_u ∝ √(σ/(1+σ)), so κ² ∝ σ/(1+σ):
   - Increasing σ by a factor c (for example by averaging c repetitions) multiplies κ² by
     c(1+σ)/(1+cσ). That is ≈ c at low SNR, and it saturates.
   - Our KU Leuven repetition averages (k = 1..4 at 5 s: 74.8, 79.6, 83.7, 86.1%) give κ
     ratios of 1.24, 1.47 and 1.62, consistent with σ ≈ 0.2–0.4.
   - MEG (κ = 0.80) against EEG (κ = 0.13–0.47) fits the same picture.
3. **Data vs SNR.** Model capacity and training data enter only through ρ*. The learning
   curve (+0.094 z per doubling of data) against the SNR curve quantifies which lever
   moves κ faster.

**Empirical test.** Numbers now live in the paper (generated into
`paper_review/aistats/tables/numbers.tex` by `tools/unified/make_numbers.py`, from
`tools/unified/snr_law.py --tag final`). This notes file is a draft of the theory
section; the paper's appendix is the authoritative version.

## Proposition 4 (content-disjoint splits; construction)
- Streams are linked when any 10-s chunk of one matches a window of another at r > 0.95,
  verified at 64 Hz.
- Groups are the connected components of the trial graph under these links.
- Folds are unions of whole groups.

**Claim.** No test stream shares a verified 10-s segment with any training or validation
stream.

**Proof.** A shared segment would create a link between a test trial and a non-test trial.
They would then lie in the same component, and so in the same fold. That contradicts the
fold being a union of whole groups. ∎

**Limits.**
- Content shared in chunks shorter than 10 s, or below the r threshold, is not detected.
- For AV-GC every trial falls into one component (stories are re-paired across subjects),
  so no content-disjoint split exists. We report it as not evaluable instead of relaxing
  the criterion.
