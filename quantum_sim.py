"""
quantum_sim.py  --  Level 3 (Bit -> Qubit), Module C (quantum simulator),
                    Module D (comparison) and the Circuit Implementation Tasks.

Uses Qiskit (circuits, Statevector) + Qiskit Aer (shot-based simulator) + NumPy.

IMPORTANT Qiskit convention: qubit 0 is the RIGHT-most character of a result string.
So the string '10' means  qubit1 = 1, qubit0 = 0.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from transistor_sim import md_table, seeing

# One shared ideal (noise-free) simulator
SIM = AerSimulator()


# ----------------------------------------------------------------------------
# Helper functions (no Streamlit here, so they are easy to test)
# ----------------------------------------------------------------------------
def build_circuit(n_qubits, init_bits, ops):
    """
    Build a circuit.
      n_qubits  : 1 or 2
      init_bits : list like [0, 1] -> start qubit0 in |0>, qubit1 in |1>
      ops       : list of tuples, e.g. ("X", 0), ("H", 1), ("Z", 0), ("CNOT", control, target)
    """
    qc = QuantumCircuit(n_qubits)
    for q, bit in enumerate(init_bits):
        if bit == 1:
            qc.x(q)                      # |1> is just an X gate applied to |0>
    if any(init_bits):
        qc.barrier()
    for op in ops:
        name = op[0]
        if name == "X":
            qc.x(op[1])
        elif name == "H":
            qc.h(op[1])
        elif name == "Z":
            qc.z(op[1])
        elif name == "CNOT":
            qc.cx(op[1], op[2])
    return qc


def get_statevector(qc):
    """Exact amplitudes of the circuit (the circuit must contain no measurements)."""
    return Statevector.from_instruction(qc)


def with_measurement(qc):
    """Return a COPY of the circuit with a measurement on every qubit."""
    m = qc.copy()
    m.measure_all()
    return m


def run_counts(qc, shots, seed=None):
    """Run the circuit `shots` times on the Aer simulator and return the counts dictionary."""
    m = with_measurement(qc)
    tqc = transpile(m, SIM, optimization_level=0)
    kwargs = {"shots": int(shots)}
    if seed is not None:
        kwargs["seed_simulator"] = int(seed)
    return SIM.run(tqc, **kwargs).result().get_counts()


def basis_labels(n_qubits):
    """['0','1'] for one qubit, ['00','01','10','11'] for two, ..."""
    return [format(i, f"0{n_qubits}b") for i in range(2 ** n_qubits)]


def plot_counts(counts, n_qubits, shots, title):
    """Probability histogram (bar chart) of measurement results."""
    labels = basis_labels(n_qubits)
    probs = [counts.get(lab, 0) / shots for lab in labels]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bars = ax.bar([f"|{lab}⟩" for lab in labels], probs, color="#2a6fbb")
    for b, p in zip(bars, probs):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.02, f"{p * 100:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Measured result")
    ax.set_ylabel("Probability (fraction of shots)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def show_circuit(qc):
    """Draw a Qiskit circuit. Falls back to text art if the pretty drawer is unavailable."""
    try:
        fig = qc.draw("mpl")
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.code(str(qc.draw("text")))


def amplitude_table(sv, n_qubits):
    """Markdown table of the statevector amplitudes and probabilities."""
    rows = []
    for lab, amp in zip(basis_labels(n_qubits), sv.data):
        amp = complex(amp)
        rows.append([f"|{lab}⟩", f"{amp.real:+.3f} {amp.imag:+.3f}i", f"{abs(amp) ** 2 * 100:.1f}%"])
    return md_table(["Basis state", "Amplitude", "Probability"], rows)


# ----------------------------------------------------------------------------
# Page: Level 3
# ----------------------------------------------------------------------------
def render_level3():
    st.header("Level 3 — Bit → Qubit")

    st.subheader("Theory")
    st.markdown(
        """
* A **classical bit** is exactly one of two values:  **b ∈ {0, 1}**.
* A **qubit** can be written as  **|ψ⟩ = α|0⟩ + β|1⟩**  where α and β are *amplitudes* and
  **|α|² + |β|² = 1**.
* When you **measure**, you still only ever read 0 or 1. The probabilities are
  **P(0) = |α|²** and **P(1) = |β|²**.
* The **Hadamard gate** makes an equal mix: **H|0⟩ = (|0⟩ + |1⟩)/√2**, so **P(0) = 50 %** and **P(1) = 50 %**.
        """
    )
    st.markdown(
        md_table(
            ["Level", "What it is", "What it can be"],
            [
                ["Transistor", "Physical switch", "ON **or** OFF"],
                ["Classical bit", "Information", "0 **or** 1"],
                ["Qubit", "Quantum state with probability amplitudes", "α|0⟩ + β|1⟩ (a blend, until measured)"],
            ],
        )
    )

    # ---------------- Optional alpha slider ----------------
    st.subheader("Optional: choose α and see P(0), P(1)")
    alpha = st.slider("Amplitude α (β is computed so that |α|² + |β|² = 1)", 0.0, 1.0, 0.71, 0.01, key="l3_alpha")
    beta = float(np.sqrt(max(0.0, 1.0 - alpha ** 2)))     # normalisation
    p0, p1 = alpha ** 2, beta ** 2
    st.write(f"α = {alpha:.2f}, β = √(1 − α²) = {beta:.2f}  →  **P(0) = {p0 * 100:.1f}%**, **P(1) = {p1 * 100:.1f}%**")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.bar(["P(0) = |α|²", "P(1) = |β|²"], [p0, p1], color=["#2a6fbb", "#d98324"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Probability")
    ax.set_title("Measurement probabilities for your α")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    seeing(
        "The two bars are the chances of reading 0 or 1. They always add up to 100 %. "
        "α = 1 is a definite 0 (like a transistor that is OFF); α ≈ 0.71 is the 50/50 Hadamard state."
    )

    # ---------------- Module C ----------------
    st.subheader("Module C — Quantum simulator")
    st.write("Pick the starting states, add gates one at a time, choose the number of shots, and look at the results.")

    n = st.radio("Number of qubits", [1, 2], horizontal=True, key="c_n")
    # If the student changes the number of qubits, start the gate list again.
    if st.session_state.get("c_last_n") != n:
        st.session_state["c_ops"] = []
        st.session_state["c_last_n"] = n
    if "c_ops" not in st.session_state:
        st.session_state["c_ops"] = []

    init_cols = st.columns(n)
    init_bits = []
    for q in range(n):
        with init_cols[q]:
            choice = st.selectbox(f"Start state of qubit {q}", ["|0⟩", "|1⟩"], key=f"c_init_{q}")
            init_bits.append(0 if choice == "|0⟩" else 1)

    gate_options = ["X", "H", "Z"] + (["CNOT"] if n == 2 else [])
    g1, g2, g3 = st.columns(3)
    with g1:
        gate = st.selectbox("Gate", gate_options, key="c_gate")
    with g2:
        if gate == "CNOT":
            control = st.selectbox("Control qubit", list(range(n)), key="c_ctrl")
        else:
            control = None
            target_single = st.selectbox("Target qubit", list(range(n)), key="c_target_single")
    with g3:
        if gate == "CNOT":
            target = st.selectbox("Target qubit", list(range(n)), index=min(1, n - 1), key="c_target")

    b1, b2, b3 = st.columns(3)
    if b1.button("Add gate", key="c_add"):
        if gate == "CNOT":
            if control == target:
                st.warning("Control and target must be different qubits.")
            else:
                st.session_state["c_ops"].append(("CNOT", control, target))
        else:
            st.session_state["c_ops"].append((gate, target_single))
    if b2.button("Undo last gate", key="c_undo") and st.session_state["c_ops"]:
        st.session_state["c_ops"].pop()
    if b3.button("Clear circuit", key="c_clear"):
        st.session_state["c_ops"] = []

    ops = st.session_state["c_ops"]
    shots = st.select_slider("Number of shots (how many times the circuit is run)",
                             options=[10, 100, 500, 1000, 5000, 10000], value=1000, key="c_shots")

    qc = build_circuit(n, init_bits, ops)
    qc_measured = with_measurement(qc)

    st.markdown("**Circuit diagram** (measurement added at the end)")
    show_circuit(qc_measured)
    seeing("Each horizontal line is a qubit, time flows left to right, boxes are gates and the "
           "meter symbols at the end are measurements. The gate list so far: "
           + (", ".join(str(o) for o in ops) if ops else "none yet (just the starting states)."))

    sv = get_statevector(qc)
    st.markdown("**Statevector amplitudes (before measurement)**")
    st.markdown(amplitude_table(sv, n))
    st.caption(f"Basis states are written |q{n - 1}…q0⟩ (qubit 0 is on the right). "
               "Probability = |amplitude|².")

    counts = run_counts(qc, shots)
    fig = plot_counts(counts, n, shots, f"Measurement results ({shots} shots)")
    st.pyplot(fig)
    plt.close(fig)
    seeing("Each bar shows how often a result appeared when the circuit was run and measured "
           f"{shots} times. With few shots the bars wobble around the true probabilities; "
           "with many shots they settle on them. Try H on |0⟩ for a 50/50 split, X on |0⟩ for a certain 1, "
           "or X then CNOT on two qubits.")
    st.markdown(
        "**Gate cheat-sheet:** X = bit flip (quantum NOT) · H = Hadamard (makes 50/50 superposition) · "
        "Z = phase flip (changes the sign of the |1⟩ amplitude; invisible to a direct measurement) · "
        "CNOT = flips the target qubit only if the control qubit is 1."
    )


# ----------------------------------------------------------------------------
# Page: Module D (comparison + live demo)
# ----------------------------------------------------------------------------
def render_module_d():
    st.header("Module D — Comparison")
    st.markdown(
        md_table(
            ["Classical — Bit", "Quantum — Qubit"],
            [
                ["0/1", "α (amplitude, i.e. α|0⟩ + β|1⟩)"],
                ["NOT", "X"],
                ["Logic operation", "Quantum gate"],
                ["Output", "Measurement"],
                ["Noise", "Quantum noise"],
                ["Deterministic", "Probabilistic measurement"],
            ],
        )
    )

    st.subheader("Live demo: deterministic vs probabilistic")
    st.write(
        "The **classical** side feeds the same input (0) through a NOT gate over and over. "
        "The **quantum** side prepares |0⟩, applies H, and measures, over and over."
    )
    runs = st.select_slider("Number of repeated runs", options=[10, 50, 100, 500, 1000, 5000], value=1000, key="d_runs")
    st.caption("Change the slider (or press R in the browser) to run the experiment again; the quantum bars change each time.")

    # Classical: NOT(0) = 1 every single time
    classical_counts = {"0": 0, "1": runs}

    # Quantum: H|0>, measure, repeated `runs` times (no fixed seed so results vary between runs)
    qc = QuantumCircuit(1)
    qc.h(0)
    quantum_counts = run_counts(qc, runs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, cnt, title, col in [
        (ax1, classical_counts, "Classical: NOT(0), same input", "#2a6fbb"),
        (ax2, quantum_counts, "Quantum: H|0⟩, same input", "#d98324"),
    ]:
        vals = [cnt.get("0", 0), cnt.get("1", 0)]
        bars = ax.bar(["Output 0", "Output 1"], vals, color=col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + runs * 0.01, str(v), ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Observed output")
        ax.grid(axis="y", alpha=0.3)
    ax1.set_ylabel(f"Number of runs (out of {runs})")
    ax1.set_ylim(0, runs * 1.15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    seeing("Left: a classical gate is deterministic – the same input gives the same output every time, "
           "so all the runs land in one bar. Right: after a Hadamard gate the measurement is "
           "probabilistic – the same input gives 0 about half the time and 1 about half the time.")


# ----------------------------------------------------------------------------
# Page: Circuit Implementation Tasks
# ----------------------------------------------------------------------------
GATE_CHOICES = ["(none)", "X", "H", "Z"]


def _ops_from_choices(choices, qubit=0):
    """Turn ['X','(none)','H'] into [('X',0), ('H',0)]."""
    return [(c, qubit) for c in choices if c != "(none)"]


def _mark_done(task_key):
    st.session_state.setdefault("tasks_done", set())
    st.session_state["tasks_done"].add(task_key)


def render_tasks():
    # Imported here (not at the top) because noise_sim itself imports this file.
    import noise_sim

    st.header("Circuit Implementation Tasks")
    st.write("Complete each task, press **Check**, and the simulator will verify your circuit.")
    st.session_state.setdefault("tasks_done", set())
    done = st.session_state["tasks_done"]
    st.progress(len(done) / 4, text=f"Tasks completed: {len(done)} / 4")

    # ---- Task 1: NOT gate ----
    with st.expander("Task 1 — Build a NOT gate (turn |0⟩ into |1⟩)", expanded=True):
        st.write("Start: one qubit in |0⟩. Choose up to three gates so that the measurement is **always 1**.")
        cols = st.columns(3)
        picks = [cols[i].selectbox(f"Gate {i + 1}", GATE_CHOICES, key=f"t1_g{i}") for i in range(3)]
        if st.button("Check Task 1", key="t1_check"):
            qc = build_circuit(1, [0], _ops_from_choices(picks))
            counts = run_counts(qc, 500, seed=1)
            show_circuit(with_measurement(qc))
            if counts.get("1", 0) == 500:
                st.success("✅ Correct! The output is 1 in all 500 shots. A single X gate is the quantum NOT.")
                _mark_done("t1")
            else:
                st.error(f"❌ Not yet. Results: {counts}. Hint: which gate flips |0⟩ to |1⟩?")

    # ---- Task 2: superposition ----
    with st.expander("Task 2 — Create a 50/50 superposition", expanded=False):
        st.write("Start: one qubit in |0⟩. Choose up to three gates so that P(0) ≈ 50 % and P(1) ≈ 50 %.")
        cols = st.columns(3)
        picks = [cols[i].selectbox(f"Gate {i + 1}", GATE_CHOICES, key=f"t2_g{i}") for i in range(3)]
        if st.button("Check Task 2", key="t2_check"):
            qc = build_circuit(1, [0], _ops_from_choices(picks))
            shots = 4000
            counts = run_counts(qc, shots, seed=2)
            p1 = counts.get("1", 0) / shots
            show_circuit(with_measurement(qc))
            fig = plot_counts(counts, 1, shots, "Your circuit")
            st.pyplot(fig)
            plt.close(fig)
            if 0.45 <= p1 <= 0.55:
                st.success(f"✅ Correct! P(1) = {p1 * 100:.1f}%, very close to 50 %.")
                _mark_done("t2")
            else:
                st.error(f"❌ Not yet: P(1) = {p1 * 100:.1f}%. Hint: use the Hadamard gate.")

    # ---- Task 3: CNOT ----
    with st.expander("Task 3 — Apply a CNOT", expanded=False):
        st.write("Start: qubit 0 = |1⟩ and qubit 1 = |0⟩. Choose up to two operations so that the final "
                 "measured result is **11** (qubit 1 and qubit 0 both 1). You must use a CNOT.")
        options = ["(none)", "X on q0", "X on q1", "H on q0", "H on q1", "CNOT q0→q1", "CNOT q1→q0"]
        cols = st.columns(2)
        picks = [cols[i].selectbox(f"Operation {i + 1}", options, key=f"t3_g{i}") for i in range(2)]
        if st.button("Check Task 3", key="t3_check"):
            mapping = {
                "X on q0": ("X", 0), "X on q1": ("X", 1), "H on q0": ("H", 0), "H on q1": ("H", 1),
                "CNOT q0→q1": ("CNOT", 0, 1), "CNOT q1→q0": ("CNOT", 1, 0),
            }
            ops = [mapping[p] for p in picks if p != "(none)"]
            qc = build_circuit(2, [1, 0], ops)
            counts = run_counts(qc, 500, seed=3)
            show_circuit(with_measurement(qc))
            used_cnot = any(o[0] == "CNOT" for o in ops)
            if used_cnot and counts.get("11", 0) == 500:
                st.success("✅ Correct! The control (q0 = 1) flipped the target, giving 11 every time.")
                _mark_done("t3")
            elif not used_cnot:
                st.error("❌ Your circuit does not contain a CNOT.")
            else:
                st.error(f"❌ Not yet. Results: {counts}. Hint: the control must be the qubit that is already 1.")

    # ---- Task 4: noisy circuit ----
    with st.expander("Task 4 — Run a noisy circuit", expanded=False):
        st.write("A qubit starts in |0⟩ and is sent through a noisy channel that flips it with probability p "
                 "(bit-flip error). Choose **p between 0.05 and 0.50**, run it, and the check passes when the "
                 "simulator shows both outcomes and the measured P(1) is close to your p.")
        p = st.slider("Bit-flip probability p", 0.0, 0.5, 0.0, 0.01, key="t4_p")
        if st.button("Run noisy circuit and check Task 4", key="t4_check"):
            shots = 2000
            counts = noise_sim.run_bit_flip(p, shots, initial_bit=0, seed=4)
            p1 = counts.get("1", 0) / shots
            fig = plot_counts(counts, 1, shots, f"Noisy circuit, p = {p:.2f}")
            st.pyplot(fig)
            plt.close(fig)
            if p < 0.05:
                st.error("❌ Choose p of at least 0.05 so that errors actually appear.")
            elif abs(p1 - p) <= 0.06 and counts.get("0", 0) > 0 and counts.get("1", 0) > 0:
                st.success(f"✅ Correct! Measured P(1) = {p1 * 100:.1f}% ≈ p = {p * 100:.0f}%. Noise made a 'sure 0' sometimes read 1.")
                _mark_done("t4")
            else:
                st.error(f"❌ Measured P(1) = {p1 * 100:.1f}%, which is not close to p. Try again.")
