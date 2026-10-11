"""
noise_sim.py  --  Noise page.

Part 1: classical transistor noise      Vout = Videal + n(t)       (NumPy)
Part 2: quantum noise                   bit flip, phase flip, depolarizing   (Qiskit Aer)
Part 3: quantum error correction        3-qubit repetition code             (Qiskit Aer)

All quantum noise is added as a Pauli error on the identity gate "id", so we place one
"id" gate in the circuit wherever we want the noisy channel to act.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from transistor_sim import md_table, seeing, VCC
from quantum_sim import plot_counts


# ----------------------------------------------------------------------------
# Part 1: classical noise (NumPy)
# ----------------------------------------------------------------------------
def q_function(x):
    """Probability that a standard Gaussian exceeds x  (tail probability)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def simulate_transistor_noise(sigma, threshold, n_bits=16, samples_per_bit=20, seed=1, vcc=VCC):
    """
    Build a noisy digital waveform.
      ideal : 0 V or Vcc for each bit
      noisy : ideal + Gaussian noise n(t) with standard deviation sigma
    Each bit is read at the MIDDLE of its time slot by comparing with the threshold.
    """
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits)
    ideal = np.repeat(bits * vcc, samples_per_bit).astype(float)
    noisy = ideal + rng.normal(0.0, sigma, ideal.size)           # Vout = Videal + n(t)
    t = np.arange(ideal.size) / samples_per_bit                  # time in "bit periods"
    mid = np.arange(n_bits) * samples_per_bit + samples_per_bit // 2
    read = (noisy[mid] > threshold).astype(int)
    return t, ideal, noisy, bits, read, mid


def theoretical_ber(sigma, threshold, vcc=VCC):
    """Expected bit-error probability for 0/1 equally likely and Gaussian noise."""
    if sigma <= 0:
        return 0.0
    p_err_given_0 = q_function(threshold / sigma)           # noise pushes a 0 above threshold
    p_err_given_1 = q_function((vcc - threshold) / sigma)   # noise pushes a 1 below threshold
    return 0.5 * (p_err_given_0 + p_err_given_1)


def monte_carlo_ber(sigma, threshold, n=20000, seed=5, vcc=VCC):
    """Measured bit-error rate over many random bits."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n)
    noisy = bits * vcc + rng.normal(0.0, sigma, n)
    return float(np.mean((noisy > threshold).astype(int) != bits))


# ----------------------------------------------------------------------------
# Part 2 + 3: quantum noise (Qiskit Aer). No Streamlit calls in these functions.
# ----------------------------------------------------------------------------
def _noisy_simulator(error):
    """An Aer simulator in which every 'id' gate suffers the given quantum error."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(error, ["id"])
    return AerSimulator(noise_model=nm)


def _run(qc, sim, shots, seed=None):
    # optimization_level=0 so the transpiler does NOT delete our 'id' gates
    tqc = transpile(qc, sim, optimization_level=0)
    kwargs = {"shots": int(shots)}
    if seed is not None:
        kwargs["seed_simulator"] = int(seed)
    return sim.run(tqc, **kwargs).result().get_counts()


def _pauli(px, py, pz):
    """Pauli error channel: X w.p. px, Y w.p. py, Z w.p. pz, otherwise nothing."""
    ops = [("X", px), ("Y", py), ("Z", pz), ("I", 1.0 - px - py - pz)]
    return pauli_error([(g, p) for g, p in ops if p > 0])


def run_bit_flip(p, shots, initial_bit=0, seed=None):
    """Qubit |initial_bit> goes through a channel that flips it (X) with probability p."""
    qc = QuantumCircuit(1, 1)
    if initial_bit == 1:
        qc.x(0)
    qc.id(0)                    # <- the noisy channel
    qc.measure(0, 0)
    return _run(qc, _noisy_simulator(_pauli(p, 0, 0)), shots, seed)


def run_phase_flip(p, shots, seed=None):
    """
    H - noisy channel (Z with probability p) - H - measure.
    Without noise the two H gates cancel and the result is always 0.
    A Z error in the middle turns into a visible bit flip after the second H.
    """
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.id(0)                    # <- the noisy channel
    qc.h(0)
    qc.measure(0, 0)
    return _run(qc, _noisy_simulator(_pauli(0, 0, p)), shots, seed)


def run_depolarizing(p, shots, seed=None):
    """|0> goes through a channel that, with probability p, applies a random X, Y or Z (p/3 each)."""
    qc = QuantumCircuit(1, 1)
    qc.id(0)                    # <- the noisy channel
    qc.measure(0, 0)
    return _run(qc, _noisy_simulator(_pauli(p / 3, p / 3, p / 3)), shots, seed)


def repetition_code_circuit(logical_bit=0):
    """
    3-qubit repetition code protecting against ONE bit flip.
      encode : copy qubit 0 onto qubits 1 and 2 with two CNOTs  ->  |b b b>
      noise  : each qubit passes through a noisy 'id' gate
      decode : two CNOTs compute the 'syndrome', a Toffoli (ccx) fixes qubit 0 by majority vote
      measure qubit 0
    """
    qc = QuantumCircuit(3, 1)
    if logical_bit == 1:
        qc.x(0)
    qc.cx(0, 1)
    qc.cx(0, 2)                 # encode
    qc.barrier()
    qc.id(0)
    qc.id(1)
    qc.id(2)                    # noise on every qubit
    qc.barrier()
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.ccx(1, 2, 0)             # decode + correct
    qc.measure(0, 0)
    return qc


def run_repetition(p, shots, logical_bit=0, seed=None):
    """Return (counts_unprotected, counts_protected) for bit-flip probability p per qubit."""
    unprotected = run_bit_flip(p, shots, logical_bit, seed)
    sim = _noisy_simulator(_pauli(p, 0, 0))
    protected = _run(repetition_code_circuit(logical_bit), sim, shots, seed)
    return unprotected, protected


def error_rate(counts, correct_bit=0):
    """Fraction of shots that were NOT the correct answer."""
    total = sum(counts.values())
    wrong = sum(v for k, v in counts.items() if int(k) != correct_bit)
    return wrong / total


@st.cache_data(show_spinner=False)
def repetition_sweep(shots=2000):
    """Error rate vs p, with and without the code (cached so sliders elsewhere stay fast)."""
    ps = [round(x, 2) for x in np.linspace(0.0, 0.5, 11)]
    un, pr = [], []
    for p in ps:
        c_un, c_pr = run_repetition(p, shots, 0, seed=123)
        un.append(error_rate(c_un))
        pr.append(error_rate(c_pr))
    return ps, un, pr


# ----------------------------------------------------------------------------
# Page: Noise
# ----------------------------------------------------------------------------
def render_noise():
    st.header("Noise")
    st.write("Real devices are never perfect. First the noise you know from electronics, then its quantum counterpart.")

    # ---------------- Part 1 ----------------
    st.subheader("1. Transistor noise:  Vout = Videal + n(t)")
    st.markdown(
        "The ideal output of a transistor stage is a clean 0 V or 5 V. In reality random noise "
        "**n(t)** is added. The receiver reads a **1** if the voltage is above a threshold and a **0** otherwise."
    )
    c1, c2 = st.columns(2)
    with c1:
        sigma = st.slider("Noise level σ (volts RMS)", 0.0, 3.0, 0.8, 0.1, key="n_sigma")
    with c2:
        thr = st.slider("Decision threshold (V)", 0.5, 4.5, 2.5, 0.1, key="n_thr")
    if "noise_seed" not in st.session_state:
        st.session_state["noise_seed"] = 1
    if st.button("Generate new random noise", key="n_reroll"):
        st.session_state["noise_seed"] += 1

    t, ideal, noisy, bits, read, mid = simulate_transistor_noise(sigma, thr, seed=st.session_state["noise_seed"])
    wrong = bits != read

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(t, ideal, color="#2a6fbb", lw=2, label="Ideal Vout")
    ax.plot(t, noisy, color="#999999", lw=1, label="Noisy Vout = ideal + n(t)")
    ax.axhline(thr, color="#d98324", ls="--", label=f"Threshold {thr:.1f} V")
    ax.plot(t[mid][~wrong], noisy[mid][~wrong], "o", color="#2e8b57", label="Bit read correctly")
    if wrong.any():
        ax.plot(t[mid][wrong], noisy[mid][wrong], "X", color="#c0392b", ms=11, label="Bit MISREAD")
    ax.set_xlabel("Time (bit periods)")
    ax.set_ylabel("Output voltage (V)")
    ax.set_title("Noisy transistor output vs ideal output")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    seeing("The blue line is the perfect signal, the grey line is what you actually measure. "
           "Dots mark the moment each bit is read. A red X means noise pushed the voltage across the "
           "orange threshold line, so a 0 was read as 1 or a 1 as 0.")

    m1, m2, m3 = st.columns(3)
    m1.metric("Misread bits (this trace)", f"{int(wrong.sum())} of {bits.size}")
    m2.metric("Measured error rate (20 000 bits)", f"{monte_carlo_ber(sigma, thr) * 100:.2f}%")
    m3.metric("Theoretical error rate", f"{theoretical_ber(sigma, thr) * 100:.2f}%")
    st.caption("Key idea: classical noise only matters if it is big enough to cross the threshold. "
               "Small noise is simply ignored – that is why digital circuits are robust.")

    # ---------------- Part 2: comparison ----------------
    st.subheader("2. Classical noise vs quantum noise")
    st.markdown(
        md_table(
            ["", "Classical (transistor) noise", "Quantum noise / decoherence"],
            [
                ["What is disturbed", "A voltage or current", "The amplitudes (and phase) of the quantum state"],
                ["Typical source", "Thermal noise, supply ripple, interference", "Unwanted coupling to the environment, imperfect gates"],
                ["Effect", "Voltage wobbles; bit flips only if the threshold is crossed", "Bit flips **and** phase flips; superposition fades into a plain 0/1 mixture (decoherence)"],
                ["Margin that protects you", "Large gap between 0 V and 5 V levels", "None for free – amplitudes are continuous, so even tiny errors matter"],
                ["Can you copy and check?", "Yes – copy the bit, use majority voting", "Not directly (no-cloning); must use entangled encoding such as the repetition code below"],
                ["Fix", "Thresholding, shielding, error-correcting codes", "Quantum error correction, better hardware, shorter circuits"],
            ],
        )
    )
    st.markdown("**Investigate: how does classical noise in transistor-based switching compare conceptually with noise affecting quantum states?**")
    st.info(
        "In both worlds noise is an unwanted random disturbance that can make the answer wrong. "
        "In a transistor circuit, information is stored in two widely separated voltage levels, so "
        "the digital threshold *cleans up* small noise and only large noise causes an error – the error is a "
        "simple bit flip. A qubit has no such clean-up: its information lives in continuous amplitudes "
        "and a relative phase, so even small disturbances accumulate, and noise can change the "
        "phase as well as the bit. Interaction with the environment also destroys superposition "
        "(*decoherence*), which has no direct classical counterpart. Finally, classical bits can be copied "
        "and majority-voted, but unknown qubit states cannot be copied, so quantum error correction "
        "needs cleverer encoding. In short: classical noise is *a voltage problem fixed by a threshold*, "
        "quantum noise is *a state problem that threatens the superposition itself*."
    )

    # ---------------- Part 3: quantum errors ----------------
    st.subheader("3. Quantum errors (Qiskit Aer simulation)")
    shots = st.select_slider("Shots for the simulations below", options=[200, 500, 1000, 2000, 5000], value=1000, key="n_shots")

    # --- Bit flip ---
    st.markdown("#### Bit-flip error")
    st.markdown("**Theory:** with probability p the X gate is applied by accident, so |0⟩ becomes |1⟩ and vice versa. "
                "This is the closest quantum cousin of a transistor bit being misread.")
    p_bf = st.slider("Bit-flip probability p", 0.0, 1.0, 0.2, 0.01, key="n_pbf")
    init = int(st.radio("Starting state", [0, 1], horizontal=True, key="n_init", format_func=lambda v: f"|{v}⟩"))
    counts = run_bit_flip(p_bf, shots, init)
    fig = plot_counts(counts, 1, shots, f"Bit-flip noise, p = {p_bf:.2f}, start |{init}⟩")
    st.pyplot(fig)
    plt.close(fig)
    seeing(f"The qubit starts as |{init}⟩, so without noise every shot would give {init}. "
           f"With bit-flip probability {p_bf:.2f}, about {p_bf * 100:.0f}% of shots give the wrong value.")

    # --- Phase flip ---
    st.markdown("#### Phase-flip error")
    st.markdown("**Theory:** with probability p a Z gate is applied, which flips the sign of the |1⟩ amplitude "
                "(α|0⟩ + β|1⟩ → α|0⟩ − β|1⟩). A phase flip is *invisible* if you measure straight away, so we "
                "use the circuit **H → noisy channel → H**. Without noise the two H gates cancel and you always read 0; "
                "a phase flip in the middle becomes a visible 1.")
    p_pf = st.slider("Phase-flip probability p", 0.0, 1.0, 0.2, 0.01, key="n_ppf")
    counts = run_phase_flip(p_pf, shots)
    fig = plot_counts(counts, 1, shots, f"Phase-flip noise, p = {p_pf:.2f}")
    st.pyplot(fig)
    plt.close(fig)
    seeing(f"The ideal answer is always 0. The share of 1s (about {p_pf * 100:.0f}%) shows how often a phase "
           "flip happened between the two Hadamard gates. This type of error does not exist in classical electronics.")

    # --- Depolarizing ---
    st.markdown("#### Depolarizing noise")
    st.markdown("**Theory:** with probability p the qubit is hit by a random error – X, Y or Z, each with probability p/3. "
                "On |0⟩, X and Y flip the bit and Z does nothing visible, so the expected P(1) = 2p/3.")
    p_dep = st.slider("Depolarizing error probability p", 0.0, 1.0, 0.3, 0.01, key="n_pdep")
    counts = run_depolarizing(p_dep, shots)
    fig = plot_counts(counts, 1, shots, f"Depolarizing noise, p = {p_dep:.2f}")
    st.pyplot(fig)
    plt.close(fig)
    seeing(f"Starting from |0⟩, the bar for 1 should be near 2p/3 = {2 * p_dep / 3 * 100:.1f}%. "
           "As p grows, information leaks away; at p = 0.75 the qubit gives 50/50 random results.")

    # ---------------- Part 4: error correction ----------------
    st.subheader("4. Quantum error correction: 3-qubit repetition code")
    st.markdown(
        "**Theory:** just like a classical majority vote, we store one logical bit in **three** qubits "
        "(|0⟩ → |000⟩, |1⟩ → |111⟩). If at most **one** qubit is flipped, the decoding step "
        "(two CNOTs + a Toffoli gate) repairs it. The code fails only if **two or more** qubits flip, "
        "which happens with probability 3p² − 2p³ – much smaller than p when p is small."
    )
    p_rep = st.slider("Bit-flip probability p on each qubit", 0.0, 0.5, 0.1, 0.01, key="n_prep")
    c_un, c_pr = run_repetition(p_rep, shots, 0)
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_counts(c_un, 1, shots, "Without error correction")
        st.pyplot(fig)
        plt.close(fig)
        st.metric("Error rate", f"{error_rate(c_un) * 100:.1f}%")
    with col2:
        fig = plot_counts(c_pr, 1, shots, "With 3-qubit repetition code")
        st.pyplot(fig)
        plt.close(fig)
        st.metric("Error rate", f"{error_rate(c_pr) * 100:.1f}%")
    seeing("Both charts start from a logical 0, so any bar at 1 is an error. The right chart (with the "
           "code) has far fewer errors than the left chart (single unprotected qubit).")

    st.markdown("**Circuit used for the protected case**")
    qc = repetition_code_circuit(0)
    try:
        fig = qc.draw("mpl")
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.code(str(qc.draw("text")))

    if st.button("Run sweep: error rate vs p (a few seconds)", key="n_sweep"):
        with st.spinner("Simulating..."):
            ps, un, pr = repetition_sweep()
        theory_p = np.linspace(0, 0.5, 101)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(ps, [u * 100 for u in un], "o-", color="#c0392b", label="No correction (simulated)")
        ax.plot(ps, [r * 100 for r in pr], "s-", color="#2a6fbb", label="3-qubit code (simulated)")
        ax.plot(theory_p, (3 * theory_p ** 2 - 2 * theory_p ** 3) * 100, "--", color="#2a6fbb", alpha=0.5, label="3p² − 2p³ (theory)")
        ax.set_xlabel("Physical bit-flip probability p per qubit")
        ax.set_ylabel("Logical error rate (%)")
        ax.set_title("Error correction helps when p is small")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        seeing("The red line (no protection) rises straight with p. The blue line (with the code) stays much lower "
               "for small p, but the two curves meet at p = 0.5 where noise is total. Error correction only "
               "pays off when individual errors are rare.")
