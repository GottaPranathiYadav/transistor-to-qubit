"""
quiz.py  --  Quiz and Assessment.

* Pre-learning assessment  (10 MCQs, shown first)
* Post-learning assessment (10 MCQs, after working through the levels)
* Instant feedback, scoring, and a pre-vs-post "Conceptual understanding" summary per level.

Scores live ONLY in st.session_state (nothing is saved to disk or a database).
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from transistor_sim import md_table, seeing

LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Noise"]


def _q(qid, level, text, options, answer, explanation):
    """Small helper to build a question dictionary. `answer` is the index of the right option."""
    return {"id": qid, "level": level, "q": text, "options": options, "answer": answer, "explanation": explanation}


QUESTIONS = {
    "pre": [
        _q("p1", "Level 1", "A transistor used as a switch has its input voltage far BELOW the switching threshold. What happens to the LED?",
           ["It is ON", "It is OFF", "It is dim and flickering", "It burns out"], 1,
           "Below the threshold the transistor is in cutoff: no current flows, so the LED is OFF (logic bit 0)."),
        _q("p2", "Level 1", "Which pairing correctly links a transistor state to a classical bit?",
           ["ON → 0, OFF → 1", "ON → 1, OFF → 0", "ON → both 0 and 1", "A transistor cannot represent a bit"], 1,
           "In the usual convention a conducting (ON) transistor represents 1 and a non-conducting (OFF) one represents 0."),
        _q("p3", "Level 2", "A transistor NOT gate receives input 1. What is its output?",
           ["0", "1", "Both", "Undefined"], 0, "A NOT gate inverts the bit: 1 → 0 and 0 → 1."),
        _q("p4", "Level 2", "Circuit task: which quantum gate flips |0⟩ to |1⟩?",
           ["Z", "H", "X", "Measurement"], 2,
           "The X gate (bit flip) is the quantum NOT: X|0⟩ = |1⟩ and X|1⟩ = |0⟩."),
        _q("p5", "Level 3", "Which expression describes a general qubit state?",
           ["b ∈ {0, 1}", "α|0⟩ + β|1⟩ with |α|² + |β|² = 1", "α + β = 1 with α, β ∈ {0, 1}", "|0⟩ × |1⟩"], 1,
           "A qubit is a combination of |0⟩ and |1⟩ with amplitudes α and β whose squared magnitudes sum to 1."),
        _q("p6", "Level 3", "Circuit task: starting from |0⟩, which single gate produces a 50 % / 50 % chance of reading 0 or 1?",
           ["X", "Z", "H", "CNOT"], 2, "H|0⟩ = (|0⟩ + |1⟩)/√2, so P(0) = P(1) = 50 %."),
        _q("p7", "Level 4", "Which sequence is the quantum counterpart of 'Transistors → Logic gates → Digital output'?",
           ["Qubits → Quantum gates → Measurement", "Qubits → Transistors → Output", "Bits → Quantum gates → Voltage", "Gates → Qubits → Resistors"], 0,
           "Quantum circuits start with qubits, apply quantum gates, and end with a measurement."),
        _q("p8", "Level 4", "Circuit task: which quantum gate plays the role of a classical XOR on the target qubit?",
           ["H", "CNOT", "Z", "X on both qubits"], 1,
           "CNOT flips the target when the control is 1, so the target becomes (target XOR control)."),
        _q("p9", "Noise", "In Vout = Videal + n(t), when is a classical bit misread?",
           ["Always, whenever n(t) is non-zero", "When the noise pushes the voltage across the decision threshold", "Only when the supply is turned off", "Never – digital signals are immune to noise"], 1,
           "Small noise is ignored by the threshold. Only noise large enough to cross it causes an error."),
        _q("p10", "Noise", "What does the 3-qubit repetition code protect against?",
           ["Any number of bit flips", "A single bit-flip error", "Only measurement", "Nothing – it adds extra noise"], 1,
           "Majority voting repairs one flipped qubit. Two or more flips defeat it."),
    ],
    "post": [
        _q("q1", "Level 1", "In the Input → Transistor → LED circuit, Vin is raised well above the threshold. What do you observe and what bit does it represent?",
           ["LED ON, bit 1", "LED OFF, bit 1", "LED ON, bit 0", "LED OFF, bit 0"], 0,
           "Above the threshold current flows and the LED lights: ON ↔ bit 1."),
        _q("q2", "Level 1", "Why does the collector/drain current stop growing at high Vin in the simulator?",
           ["The transistor melts", "The supply, LED and load resistor limit the current (saturation)", "Vin is reset to 0", "The threshold moves"], 1,
           "In saturation the external circuit, I ≈ (Vcc − Vf)/RL, sets the maximum current."),
        _q("q3", "Level 2", "Which statement about the X gate is correct?",
           ["It is analogous to a quantum NOT: X|0⟩ = |1⟩ and X|1⟩ = |0⟩", "It creates a 50/50 superposition", "It measures the qubit", "It changes only the phase"], 0,
           "X is the bit-flip gate, the quantum equivalent of NOT."),
        _q("q4", "Level 2", "Circuit task: you want a qubit that starts in |1⟩ to end in |0⟩. Which gate do you apply?",
           ["H", "Z", "X", "No gate is needed"], 2, "X|1⟩ = |0⟩."),
        _q("q5", "Level 3", "A qubit has α = 0.6. What is the magnitude of β and the probability of measuring 1?",
           ["β = 0.4, P(1) = 40 %", "β = 0.8, P(1) = 64 %", "β = 0.6, P(1) = 36 %", "β = 1, P(1) = 100 %"], 1,
           "|β|² = 1 − 0.36 = 0.64, so β = 0.8 and P(1) = 64 %."),
        _q("q6", "Level 3", "Circuit task: you apply H to |0⟩ and measure 1000 times. What do you expect?",
           ["Always 0", "Always 1", "About 500 zeros and 500 ones, varying each time", "Exactly 500 and 500 every time"], 2,
           "Measurement of a superposition is probabilistic, so the counts hover around 50/50 but are not identical each run."),
        _q("q7", "Level 4", "For Y = (NOT A) XOR B implemented with qubits, which gates are used?",
           ["X on A then CNOT(A → B), then measure B", "H on both, then measure", "Z on A and Z on B", "Two measurements and no gates"], 0,
           "X gives NOT A, then CNOT writes (NOT A) XOR B onto the target qubit, which is measured."),
        _q("q8", "Level 4", "How does the classical NOT/XOR circuit differ from its quantum version for basis-state inputs?",
           ["The quantum version gives different answers", "Both give the same deterministic answer", "The classical version is probabilistic", "Neither gives an answer"], 1,
           "X and CNOT map basis states to basis states, so the results match the classical gates."),
        _q("q9", "Noise", "Which error exists for qubits but has no direct classical equivalent?",
           ["Bit flip", "Phase flip (changing the sign of the |1⟩ amplitude)", "Misreading a voltage", "Resistor tolerance"], 1,
           "Phase information is quantum-only; classical bits have no phase."),
        _q("q10", "Noise", "With bit-flip probability p = 0.1 per qubit, why does the 3-qubit repetition code help?",
           ["It stops errors from ever occurring", "It only fails when two or more qubits flip, which is about 3p² ≈ 3 %", "It measures the qubit twice", "It removes the need for measurement"], 1,
           "The logical error probability is 3p² − 2p³ ≈ 2.8 % for p = 0.1, much lower than 10 %."),
    ],
}


# ----------------------------------------------------------------------------
# Scoring helpers (no Streamlit calls, easy to test)
# ----------------------------------------------------------------------------
def level_scores(results, phase):
    """
    results[phase] = {qid: True/False}.
    Return {level: percent_correct} using only the questions of that phase.
    """
    out = {}
    for level in LEVELS:
        qs = [q for q in QUESTIONS[phase] if q["level"] == level]
        correct = sum(1 for q in qs if results[phase].get(q["id"]) is True)
        out[level] = 100.0 * correct / len(qs) if qs else 0.0
    return out


def total_score(results, phase):
    """(correct, total) for a phase."""
    correct = sum(1 for v in results[phase].values() if v)
    return correct, len(QUESTIONS[phase])


def phase_complete(results, phase):
    return len(results[phase]) == len(QUESTIONS[phase])


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
def _render_phase(phase, title):
    results = st.session_state["quiz_results"]
    st.subheader(title)
    for i, q in enumerate(QUESTIONS[phase], start=1):
        qid = q["id"]
        answered = qid in results[phase]
        with st.container(border=True):
            st.markdown(f"**Q{i}. ({q['level']})** {q['q']}")
            choice = st.radio("Your answer", q["options"], index=None, key=f"{phase}_{qid}_choice",
                              disabled=answered, label_visibility="collapsed")
            if not answered:
                if st.button("Check answer", key=f"{phase}_{qid}_btn"):
                    if choice is None:
                        st.warning("Pick an option first.")
                    else:
                        results[phase][qid] = (q["options"].index(choice) == q["answer"])
                        st.rerun()
            else:
                if results[phase][qid]:
                    st.success("✅ Correct. " + q["explanation"])
                else:
                    st.error("❌ Not quite. The right answer is: **" + q["options"][q["answer"]] + "**. " + q["explanation"])

    c, n = total_score(results, phase)
    st.markdown(f"**Score so far: {c} / {n}** ({len(results[phase])} answered)")
    if phase_complete(results, phase):
        st.success(f"{title} finished: {c} / {n}.")


def _render_summary():
    results = st.session_state["quiz_results"]
    st.subheader("Conceptual understanding summary")
    if not (phase_complete(results, "pre") and phase_complete(results, "post")):
        st.info("Finish both the pre-learning and post-learning assessments to see the full comparison.")
        if results["pre"] or results["post"]:
            st.write(f"Pre-learning so far: {total_score(results, 'pre')[0]} correct of {len(results['pre'])} answered. "
                     f"Post-learning so far: {total_score(results, 'post')[0]} correct of {len(results['post'])} answered.")
        return

    pre = level_scores(results, "pre")
    post = level_scores(results, "post")
    pre_c, pre_n = total_score(results, "pre")
    post_c, post_n = total_score(results, "post")
    pre_pct, post_pct = 100.0 * pre_c / pre_n, 100.0 * post_c / post_n

    m1, m2, m3 = st.columns(3)
    m1.metric("Pre-learning score", f"{pre_c}/{pre_n} ({pre_pct:.0f}%)")
    m2.metric("Post-learning score", f"{post_c}/{post_n} ({post_pct:.0f}%)")
    if pre_pct > 0:
        m3.metric("Improvement", f"{(post_pct - pre_pct) / pre_pct * 100:+.0f}%", delta=f"{post_pct - pre_pct:+.0f} percentage points")
    else:
        m3.metric("Improvement", "n/a (pre score was 0)", delta=f"{post_pct - pre_pct:+.0f} percentage points")

    x = np.arange(len(LEVELS))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.2, [pre[l] for l in LEVELS], 0.4, label="Pre-learning", color="#999999")
    ax.bar(x + 0.2, [post[l] for l in LEVELS], 0.4, label="Post-learning", color="#2a6fbb")
    ax.set_xticks(x)
    ax.set_xticklabels(LEVELS)
    ax.set_ylim(0, 115)
    ax.set_xlabel("Framework level")
    ax.set_ylabel("Questions answered correctly (%)")
    ax.set_title("Conceptual understanding: pre vs post, per level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    seeing("Grey bars are your pre-learning score for each level and blue bars are your post-learning score. "
           "Taller blue bars mean you understood that level better after working through the site. "
           "Each level has two questions, so each bar can only be 0 %, 50 % or 100 %.")

    rows = []
    for l in LEVELS:
        change = post[l] - pre[l]
        rel = f"{change / pre[l] * 100:+.0f}%" if pre[l] > 0 else "n/a"
        rows.append([l, f"{pre[l]:.0f}%", f"{post[l]:.0f}%", f"{change:+.0f} pp", rel])
    st.markdown(md_table(["Level", "Pre", "Post", "Change", "% improvement"], rows))


def render_quiz():
    st.header("Quiz and Assessment")
    # Scores are kept in session state only.
    if "quiz_results" not in st.session_state:
        st.session_state["quiz_results"] = {"pre": {}, "post": {}}
    results = st.session_state["quiz_results"]

    st.write("Take the **pre-learning** assessment before you study the levels, and the **post-learning** "
             "assessment after. Your scores are kept only for this browser session.")
    view = st.radio("Section", ["Pre-learning assessment", "Post-learning assessment", "Conceptual understanding summary"],
                    horizontal=True, key="quiz_view")

    if view == "Pre-learning assessment":
        _render_phase("pre", "Pre-learning assessment")
    elif view == "Post-learning assessment":
        if not phase_complete(results, "pre"):
            st.info("Please finish the pre-learning assessment first, then work through Levels 1–4, Module D and Noise, "
                    "and come back here for the post-learning assessment.")
        else:
            _render_phase("post", "Post-learning assessment")
    else:
        _render_summary()
