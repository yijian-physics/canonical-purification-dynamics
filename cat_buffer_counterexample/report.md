# A failed buffered update from the exact canonical cat state

The finite-buffer trace-polar Kraus update fails for a local noise channel acting on a coherent logical sector. The earlier stationarity counterexample started in a wrong gauge. This example starts on the exact positive branch.

Let P₊ and P₋ project onto the normalized GHZ± cats. Start with Q₀=P₊ and apply the channel Φ(X)=(1−p)X+p ZⱼXZⱼ to one spin. Since Zⱼ exchanges the cats, the exact density and amplitude are

$$
\rho'=(1-p)P_+ + pP_-,\qquad \sqrt{\rho'}=\sqrt{1-p}\,P_+ + \sqrt p\,P_-.
$$

For any proper window W containing j, tracing out even one spin removes the off-diagonal terms between the two ferromagnetic strings. Thus A=Tr_out Q₀ is half the projector S onto the two uniform strings in W. It commutes with Zⱼ. Write a=√(1−p), b=√p. The local polar objective uses C=[aA,bZⱼA]. Because CC†=A², its two reference blocks on the support are aS and bZⱼS. Inserting them into the update gives

$$
Q_{\rm raw}=(1-p)Q_0+pZ_jQ_0Z_j=\rho',\qquad
Q_{\rm alg}=\frac{\rho'}{\sqrt{(1-p)^2+p^2}}.
$$

The result is positive. Trace optimization is already stationary. Its represented density assigns probability p²/[(1−p)²+p²] to the minus cat. The unnormalized squared-norm loss is 2p(1−p), so the physical-loss check can flag the step. Successive proper buffers all give the same incorrect answer.

For repeated applications, the amplitude stays in the positive span of P₊ and P₋. Its partial trace is again a positive scalar times S, and each raw update again applies Φ to the amplitude. Intermediate normalization only rescales the result. Any subdivision of the same dephasing channel therefore gives the same final normalized amplitude.

At p=0.1 we found a minus-sector probability of 0.0121951 versus the exact 0.1, and a global amplitude error of 0.210702. For the two opposite input cats, the computed overlap was 0.219512 versus the exact unsquared Holevo affinity 2√[p(1−p)]=0.6. Tests on 6, 16 and 64 spins reproduced the formulas at bond four with zero reported spatial truncation. On 64 spins we also subdivided the channel into 32 steps. A six-spin whole-chain window reproduced the exact root within 3×10⁻¹⁶. At p=0.1 the single-step loss was 0.18 and the existing loss-sensitive acceptance check rejected the update at tolerance 10⁻⁴.

This example probes long-range Markov structure. For the pure initial GHZ state, any tripartition into nonempty exterior, buffer and target covering the chain has conditional mutual information of one bit. A guarantee conditional on a sufficiently small full-exterior CMI would have different hypotheses. For this cat channel, propagating the two logical probabilities and taking their square roots gives the exact global amplitude and affinity at every size.

The regression also exposed a separate implementation defect. When the orthogonality center moved right, the sweep could cache a prefix contraction before the QR step changed the last tensor in that prefix. This could trigger spurious reference rotations even on a positive amplitude. The corrected implementation moves the center first. The cat result above holds before any reference sweep and after correcting this defect. The existing local validation passed. Rechecks of the four-spin 10-step buffer-two, six-spin 10-step buffer-two, and six-spin 32-step buffer-three Ising controls reproduced the previously reported CP errors at their quoted precision. Other archive rows are historical.

Run the accompanying cat_buffer_counterexample.py with NumPy and SciPy to reproduce the measurements.
