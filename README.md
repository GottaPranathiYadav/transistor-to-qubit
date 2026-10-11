# Transistor-to-Qubit Learning Framework

An educational Streamlit website that takes electronics students from the **transistor** (ON/OFF switch)
to the **qubit** (superposition, gates, measurement, noise) using Python, NumPy, Qiskit and Qiskit Aer.

## File structure

```
t2q/
├── app.py             # navigation (sidebar) + Home page + How-to-Run page
├── transistor_sim.py  # Level 1: transistor model, hardware build guide, Module A
├── logic_sim.py       # Level 2: logic gates (Module B), NOT vs X;  Level 4: classical vs quantum
├── quantum_sim.py     # Level 3: qubit simulator (Module C), Module D comparison, circuit tasks
├── noise_sim.py       # Noise: transistor noise, bit/phase flip, depolarizing, repetition code
├── quiz.py            # Pre/post assessments and conceptual-understanding summary
├── requirements.txt
└── README.md
```

## Install

Python 3.9 or newer is recommended. In a terminal, inside the project folder:

```bash
pip install streamlit numpy qiskit qiskit-aer matplotlib
pip install pylatexenc      # only needed for the pretty circuit drawings (text drawings are used if it is missing)
```

(or simply `pip install -r requirements.txt`)

## Run

```bash
streamlit run app.py
```

Your browser opens at http://localhost:8501. Use the sidebar to move between pages.

**Suggested study order:** take the pre-learning assessment (page 7) → pages 1 to 6 → circuit tasks (page 8) →
post-learning assessment (page 7) → read the conceptual-understanding summary.

## Reproducing circuits in IBM Quantum Composer

1. Open IBM Quantum Composer: <https://quantum.cloud.ibm.com/composer> (free IBM account; search "IBM Quantum Composer" if the link has moved).
2. Create a new circuit. Keep the qubit lines you need (1 or 2). Drag gates from the palette onto the lines, left to right in time order.
3. Add a **Measure** operation at the end of each qubit, then run the simulator and look at the probability histogram.

Recipes for the circuits used on this site (qubit 0 is the top line, `q[0]`):

| Circuit | Composer steps | Expected result |
|---|---|---|
| NOT gate (X) | 1 qubit: place **X** on q[0], then Measure | Always `1` |
| 50/50 superposition | 1 qubit: place **H** on q[0], then Measure | About 50 % `0`, 50 % `1` |
| Z gate | 1 qubit: **H**, **Z**, **H**, Measure | Always `1` (the Z phase flip becomes visible) |
| CNOT | 2 qubits: **X** on q[0], then **CNOT** (control q[0], target q[1]), Measure both | Always `11` |
| Level 4: Y = (NOT A) XOR B | 2 qubits: set A and B (X on a qubit makes it 1), **X** on q[0], **CNOT** control q[0] → target q[1], Measure | Output Y is q[1] |

Note: Qiskit prints results with qubit 0 on the **right** (`q1 q0`), and Composer may show it the same way.

## Notes

* Scores in the quiz are stored only in Streamlit session state (they disappear when you close the browser tab).
* The transistor model is a simple NumPy square-law switch intended for intuition, not circuit design.
* Quantum noise is simulated with Qiskit Aer by adding Pauli errors to an identity (`id`) gate.
* The hardware build guide (BJT/MOSFET, breadboard, resistor, LED, Arduino/ESP32) is on the Level 1 page.
