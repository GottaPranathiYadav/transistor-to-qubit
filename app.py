"""
app.py  --  Navigation and home page of the "Transistor-to-Qubit Learning Framework".

Run with:   streamlit run app.py

Each page is drawn by a function that lives in its own module:
    transistor_sim.py -> Level 1
    logic_sim.py      -> Level 2 and Level 4
    quantum_sim.py    -> Level 3, Module D, Circuit Implementation Tasks
    noise_sim.py      -> Noise
    quiz.py           -> Quiz and Assessment
"""

import os
import streamlit as st

st.set_page_config(page_title="Transistor-to-Qubit Learning Framework", page_icon="🔌", layout="wide")

# The quantum pages need Qiskit. Show a friendly message if it is missing.
try:
    import transistor_sim
    import logic_sim
    import quantum_sim
    import noise_sim
    import quiz
except ImportError as err:
    st.error(f"A required package is missing: {err}")
    st.code("pip install streamlit numpy qiskit qiskit-aer matplotlib pylatexenc", language="bash")
    st.stop()

from transistor_sim import md_table

PAGES = [
    "0. Home / Overview",
    "1. Level 1 — Transistor",
    "2. Level 2 — Logic Gate",
    "3. Level 3 — Bit → Qubit",
    "4. Level 4 — Classical vs Quantum Circuit",
    "5. Module D — Comparison",
    "6. Noise",
    "7. Quiz and Assessment",
    "8. Circuit Implementation Tasks",
    "9. How to Run",
]


def render_home():
    st.title("Transistor-to-Qubit Learning Framework")
    st.caption("From Transistors to Qubits: A Hybrid Learning Framework for Understanding Quantum "
               "Computing through Semiconductor Concepts")
    st.markdown(
        """
If you know **transistors** and **digital logic**, you already know more about quantum computing than
you think. But there is a conceptual gap: a transistor is *either ON or OFF*, a classical bit is *either
0 or 1*, whereas a **qubit** can be in a *blend* of 0 and 1 described by probability amplitudes, and quantum
gates, measurement and noise behave differently from their electronic cousins.

This site bridges the gap in **four levels**. You start with the electronics you know and move step by step to
Qiskit and quantum circuits:

1. **Transistor** – a switch that turns an LED ON/OFF → the classical bit.
2. **Logic gate** – the transistor NOT gate → the quantum X gate.
3. **Bit → Qubit** – superposition, amplitudes and the Hadamard gate.
4. **Classical vs quantum circuit** – the same function built both ways.

Then compare the two worlds (Module D), meet **noise** and error correction, and test yourself with quizzes and circuit tasks.
        """
    )
    st.info("👉 **Start here:** open **7. Quiz and Assessment** and take the *pre-learning* assessment first, "
            "then work through pages 1 → 6, do the circuit tasks (page 8), and finish with the *post-learning* assessment.")

    st.subheader("Classical-to-Quantum mapping")
    st.markdown(
        md_table(
            ["Electronics concept", "Student already knows", "Quantum counterpart"],
            [
                ["Transistor", "Electronic switching device", "Qubit as quantum information unit"],
                ["ON/OFF", "0/1", "Measurement gives 0 or 1"],
                ["Voltage/current state", "Physical representation of information", "Quantum state/amplitude"],
                ["Logic gate", "AND, OR, NOT", "X, Y, Z, H, CNOT gates"],
                ["NOT gate", "Inverts 0↔1", "X gate performs quantum bit flip"],
                ["Digital circuit", "Combination of logic gates", "Quantum circuit"],
                ["Noise", "Disturbs transistor signal", "Decoherence/quantum noise"],
                ["Measurement", "Reading circuit output", "Quantum measurement"],
            ],
        )
    )


FALLBACK_README = """
**Install:** `pip install streamlit numpy qiskit qiskit-aer matplotlib pylatexenc`

**Run:** `streamlit run app.py`

README.md was not found next to app.py. See the project folder for the full instructions.
"""


def render_how_to_run():
    st.header("How to Run")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    try:
        with open(path, encoding="utf-8") as f:
            st.markdown(f.read())
    except OSError:
        st.markdown(FALLBACK_README)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🔌 → ⚛️ Navigation")
page = st.sidebar.radio("Go to page", PAGES, key="nav_page")
st.sidebar.caption("Tip: all scores are kept only in this browser session.")

if page == PAGES[0]:
    render_home()
elif page == PAGES[1]:
    transistor_sim.render_level1()
elif page == PAGES[2]:
    logic_sim.render_level2()
elif page == PAGES[3]:
    quantum_sim.render_level3()
elif page == PAGES[4]:
    logic_sim.render_level4()
elif page == PAGES[5]:
    quantum_sim.render_module_d()
elif page == PAGES[6]:
    noise_sim.render_noise()
elif page == PAGES[7]:
    quiz.render_quiz()
elif page == PAGES[8]:
    quantum_sim.render_tasks()
else:
    render_how_to_run()
