"""
logic_sim.py  --  Level 2 (Logic Gate), Module B (logic-gate simulator),
                  and Level 4 (Classical circuit vs Quantum circuit).
"""

import matplotlib.pyplot as plt
import streamlit as st

from transistor_sim import md_table, seeing
from quantum_sim import build_circuit, run_counts, show_circuit, with_measurement

# ----------------------------------------------------------------------------
# Classical gates: name -> (number of inputs, function, plain-English description)
# ----------------------------------------------------------------------------
GATES = {
    "NOT":  (1, lambda a, b=0: 1 - a,           "Output is the opposite of the input. (1 transistor + 1 resistor)"),
    "AND":  (2, lambda a, b: a & b,             "Output is 1 only when BOTH inputs are 1. (transistors in series)"),
    "OR":   (2, lambda a, b: a | b,             "Output is 1 when AT LEAST ONE input is 1. (transistors in parallel)"),
    "NAND": (2, lambda a, b: 1 - (a & b),       "AND followed by NOT: output is 0 only when both inputs are 1."),
    "NOR":  (2, lambda a, b: 1 - (a | b),       "OR followed by NOT: output is 1 only when both inputs are 0."),
    "XOR":  (2, lambda a, b: a ^ b,             "Output is 1 when the inputs are DIFFERENT."),
}


def gate_output(name, a, b=0):
    """Compute a gate output (0 or 1)."""
    return int(GATES[name][1](a, b))


def truth_table(name):
    """List of rows [A, B, Y] (or [A, Y] for NOT)."""
    if GATES[name][0] == 1:
        return [[a, gate_output(name, a)] for a in (0, 1)]
    return [[a, b, gate_output(name, a, b)] for a in (0, 1) for b in (0, 1)]


def gate_diagram(name):
    """A simple text diagram of the gate."""
    if GATES[name][0] == 1:
        return "A ──▷○── Y        (NOT: triangle with a bubble)"
    return f"A ──┐\n    ├─[ {name} ]── Y\nB ──┘"


# ----------------------------------------------------------------------------
# Page: Level 2 (+ Module B)
# ----------------------------------------------------------------------------
def render_level2():
    st.header("Level 2 — Logic Gate")

    st.subheader("Theory")
    st.markdown(
        """
**Transistor NOT gate (inverter):** input 0 → output 1, and input 1 → output 0.

| Input | Output |
|---|---|
| 0 | 1 |
| 1 | 0 |

**Quantum equivalent – the X gate:**

* X|0⟩ = |1⟩
* X|1⟩ = |0⟩

The **X gate is analogous to a quantum NOT operation** (it is also called the *bit-flip* gate).
        """
    )

    # ---------------- Module B ----------------
    st.subheader("Module B — Logic-gate simulator")
    name = st.selectbox("Choose a gate", list(GATES.keys()), index=0, key="b_gate")
    n_inputs, _, desc = GATES[name]

    c1, c2 = st.columns(2)
    with c1:
        a = int(st.radio("Input A", [0, 1], horizontal=True, key="b_a"))
    with c2:
        if n_inputs == 2:
            b = int(st.radio("Input B", [0, 1], horizontal=True, key="b_b"))
        else:
            b = 0
            st.caption("NOT has only one input.")

    y = gate_output(name, a, b)
    st.metric(f"Output of {name}", y)
    st.markdown(f"**How it works:** {desc}")
    st.code(gate_diagram(name), language=None)

    st.markdown("**Truth table**")
    if n_inputs == 1:
        st.markdown(md_table(["A", "Y"], truth_table(name)))
    else:
        st.markdown(md_table(["A", "B", "Y"], truth_table(name)))

    # ---------------- NOT vs X ----------------
    st.subheader("NOT gate vs X gate — side by side")
    bit = int(st.radio("Input bit / starting state", [0, 1], horizontal=True, key="b_notx"))
    left, right = st.columns(2)
    with left:
        st.markdown("**Classical NOT (transistor inverter)**")
        st.write(f"Input = {bit}  →  Output = **{gate_output('NOT', bit)}**")
        st.code("A ──▷○── Y", language=None)
    with right:
        st.markdown("**Quantum X gate (Qiskit)**")
        qc = build_circuit(1, [bit], [("X", 0)])
        show_circuit(with_measurement(qc))
        counts = run_counts(qc, 100, seed=7)
        result = max(counts, key=counts.get)
        st.write(f"Start |{bit}⟩  →  measured **{result}**  (all 100 shots: {counts})")
    seeing("Both sides flip the bit. Left: the classical inverter. Right: the quantum X gate "
           "applied to a qubit that starts as |0⟩ or |1⟩ (shown in the circuit, the X box is the gate "
           "and the meter at the end is the measurement). For basis states the results are identical – "
           "the difference only appears once superposition is involved (Level 3).")


# ----------------------------------------------------------------------------
# Page: Level 4
# ----------------------------------------------------------------------------
def quantum_not_xor(a, b, shots=100, seed=11):
    """
    Quantum version of  Y = (NOT A) XOR B.
      qubit0 = A, qubit1 = B.
      X on qubit0           -> qubit0 = NOT A
      CNOT(control 0, target 1) -> qubit1 = B XOR (NOT A)
    The output Y is read from qubit1 (the LEFT character of the Qiskit result string).
    Returns (y, counts).
    """
    qc = build_circuit(2, [a, b], [("X", 0), ("CNOT", 0, 1)])
    counts = run_counts(qc, shots, seed=seed)
    top = max(counts, key=counts.get)
    return int(top[0]), counts, qc


def render_level4():
    st.header("Level 4 — Classical Circuit vs Quantum Circuit")
    st.markdown(
        """
We implement the **same logical function** both ways:

> **Y = (NOT A) XOR B**   — it uses a NOT gate and a two-input XOR gate (CNOT in the quantum version).

* **Classical flow:** Transistors → Logic gates → Digital output
* **Quantum flow:** Qubits → Quantum gates → Measurement
        """
    )
    c1, c2 = st.columns(2)
    with c1:
        a = int(st.radio("Input A", [0, 1], horizontal=True, key="l4_a"))
    with c2:
        b = int(st.radio("Input B", [0, 1], horizontal=True, key="l4_b"))

    # Classical result
    not_a = gate_output("NOT", a)
    y_classical = gate_output("XOR", not_a, b)

    # Quantum result
    y_quantum, counts, qc = quantum_not_xor(a, b)

    left, right = st.columns(2)
    with left:
        st.subheader("Classical")
        st.markdown("Transistors → Logic gates → Digital output")
        st.code(f"A = {a} ──▷○── NOT A = {not_a} ──┐\n"
                f"                           ├─[ XOR ]── Y = {y_classical}\n"
                f"B = {b} ───────────────────┘", language=None)
        st.metric("Digital output Y", y_classical)
    with right:
        st.subheader("Quantum")
        st.markdown("Qubits → Quantum gates → Measurement")
        show_circuit(with_measurement(qc))
        st.write(f"Raw measurement counts (|q1 q0⟩, 100 shots): {counts}")
        st.metric("Measured output Y (qubit 1)", y_quantum)

    if y_classical == y_quantum:
        st.success("Both approaches give the same answer.")
    else:
        st.error("The results differ – this should not happen.")

    # Truth tables for all inputs
    st.markdown("**All inputs, both methods**")
    rows = []
    for aa in (0, 1):
        for bb in (0, 1):
            yc = gate_output("XOR", gate_output("NOT", aa), bb)
            yq, _, _ = quantum_not_xor(aa, bb)
            rows.append([aa, bb, yc, yq, "✔" if yc == yq else "✘"])
    st.markdown(md_table(["A", "B", "Classical Y", "Quantum Y", "Match"], rows))

    # A small bar chart of the quantum counts for the chosen input
    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["00", "01", "10", "11"]
    ax.bar([f"|{l}⟩" for l in labels], [counts.get(l, 0) for l in labels], color="#2a6fbb")
    ax.set_xlabel("Measured |q1 q0⟩")
    ax.set_ylabel("Number of shots (out of 100)")
    ax.set_title("Quantum circuit measurement counts")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    seeing("For the chosen A and B the quantum circuit gives one single result in all 100 shots, "
           "just like the classical gate. That is because X and CNOT only move basis states "
           "around; no superposition is created here. The output Y is the left digit of the result.")
