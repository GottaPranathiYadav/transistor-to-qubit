"""
transistor_sim.py  --  Level 1: the Transistor  (Module A)

A transistor used as a switch:   Input (Vin) --> Transistor --> LED
  * Vin below the threshold  -> transistor OFF (cutoff)   -> no current -> LED OFF -> bit 0
  * Vin well above threshold -> transistor ON (saturation) -> current     -> LED ON  -> bit 1

The physics here is a deliberately SIMPLE NumPy model (a "square-law" MOSFET-like
switch).  It is meant for intuition, not for designing real circuits.

This file also holds two tiny helpers (md_table, seeing) that every other page re-uses.
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------------
# Fixed circuit constants (the sliders change the other values)
# ----------------------------------------------------------------------------
VCC = 5.0            # supply voltage in volts
LED_VF = 2.0         # LED forward voltage drop in volts (typical red LED)
K = 2e-3             # transistor gain constant in A/V^2 (how strongly Vin turns it on)
LED_ON_CURRENT = 1e-3  # we call the LED "ON" once it carries at least 1 mA


# ----------------------------------------------------------------------------
# Shared UI helpers (used by all pages)
# ----------------------------------------------------------------------------
def md_table(headers, rows):
    """Return a Markdown table string (so we get a clean table without an index column)."""
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def seeing(text):
    """The 'What am I seeing?' caption that goes under every graph."""
    st.caption("**What am I seeing?** " + text)


# ----------------------------------------------------------------------------
# The NumPy transistor model
# ----------------------------------------------------------------------------
def drain_current(vin, vth, rl, vcc=VCC, vf=LED_VF, k=K):
    """
    Collector/drain current (in amperes) for a given input voltage.

    cutoff     : Vin <= Vth                 -> I = 0
    active     : Vth < Vin, current grows   -> I = k * (Vin - Vth)^2
    saturation : current is capped by the external circuit:  I_max = (Vcc - Vf) / RL
    """
    vin = np.asarray(vin, dtype=float)
    i_max = max(vcc - vf, 0.0) / rl                       # limit set by supply, LED and resistor
    i_active = np.where(vin > vth, k * (vin - vth) ** 2, 0.0)
    return np.minimum(i_active, i_max)


def region(vin, vth, rl, vcc=VCC, vf=LED_VF, k=K):
    """Name the operating region: 'Cutoff', 'Active' or 'Saturation'."""
    if vin <= vth:
        return "Cutoff"
    i_max = max(vcc - vf, 0.0) / rl
    if k * (vin - vth) ** 2 >= i_max:
        return "Saturation"
    return "Active"


def drain_voltage(i_d, rl, vcc=VCC, vf=LED_VF):
    """Voltage at the transistor's collector/drain pin (the 'output' of the stage)."""
    i_d = np.asarray(i_d, dtype=float)
    return vcc - i_d * rl - np.where(i_d > 0, vf, 0.0)


def led_is_on(i_d):
    """LED is ON when the current is at least LED_ON_CURRENT."""
    return bool(i_d >= LED_ON_CURRENT)


# ----------------------------------------------------------------------------
# Page: Level 1
# ----------------------------------------------------------------------------
def render_level1():
    st.header("Level 1 — Transistor")

    st.subheader("Theory")
    st.markdown(
        """
A transistor can be used as an **electronic switch**. A small control voltage at the
base/gate decides whether a larger current can flow through the LED:

```
Input (Vin)  -->  Transistor  -->  LED
```

* **Vin = 0** → transistor OFF → no current → LED **OFF**
* **Vin = 1** (a "high" voltage) → transistor ON → current flows → LED **ON**

So the two physical states ON/OFF are used to represent the two values of a
**classical bit: 0 or 1**.
        """
    )

   

    # ---------------- Module A ----------------
    st.subheader("Module A — Transistor simulator")
    st.write(
        f"Circuit: supply = {VCC:.0f} V, LED drop = {LED_VF:.0f} V. "
        "Move the sliders and watch the LED, the current and the logic bit."
    )

    c1, c2 = st.columns(2)
    with c1:
        vin = st.slider("Base/gate voltage Vin (V)", 0.0, 5.0, 0.0, 0.1, key="l1_vin")
        vth = st.slider("Switching threshold Vth (V)", 0.5, 4.0, 2.0, 0.1, key="l1_vth")
    with c2:
        rl = st.slider("Load resistance RL (Ω)", 100, 2000, 330, 10, key="l1_rl")

    i_d = float(drain_current(vin, vth, rl))
    v_out = float(drain_voltage(i_d, rl))
    on = led_is_on(i_d)
    reg = region(vin, vth, rl)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Collector/drain current", f"{i_d * 1000:.2f} mA")
    m2.metric("Output voltage (collector/drain)", f"{v_out:.2f} V")
    m3.metric("Operating region", reg)
    m4.metric("Logic bit", "1" if on else "0")
    if on:
        st.success("💡 LED is ON  →  logic bit = 1")
    else:
        st.info("⚫ LED is OFF  →  logic bit = 0")

    # --- the switching curve ---
    v_axis = np.linspace(0, VCC, 300)
    i_curve = drain_current(v_axis, vth, rl) * 1000.0     # mA
    vout_curve = drain_voltage(drain_current(v_axis, vth, rl), rl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(v_axis, i_curve, color="#2a6fbb", lw=2, label="Current")
    ax1.axvline(vth, color="gray", ls="--", label=f"Threshold = {vth:.1f} V")
    ax1.axhline(LED_ON_CURRENT * 1000, color="#d98324", ls=":", label="LED-on current (1 mA)")
    ax1.plot([vin], [i_d * 1000], "o", color="#c0392b", ms=9, label="You are here")
    ax1.set_xlabel("Input voltage Vin (V)")
    ax1.set_ylabel("Collector/drain current (mA)")
    ax1.set_title("Switching curve: current vs input voltage")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(v_axis, vout_curve, color="#2a6fbb", lw=2)
    ax2.axvline(vth, color="gray", ls="--")
    ax2.plot([vin], [v_out], "o", color="#c0392b", ms=9)
    ax2.set_xlabel("Input voltage Vin (V)")
    ax2.set_ylabel("Collector/drain voltage (V)")
    ax2.set_title("Output voltage vs input voltage")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    seeing(
        "Left: current through the LED as Vin rises. Nothing flows until Vin passes the threshold "
        "(dashed line), then the current climbs and finally flattens (saturation) because the "
        "resistor and supply limit it. The red dot is your slider setting. Right: the voltage at "
        "the collector/drain falls from 5 V toward 0 V as the transistor turns ON."
    )
    st.markdown(
        "**Try it:** raise Vin slowly past the threshold – the LED turns ON and the bit flips "
        "from 0 to 1. Lower the threshold or the load resistance to see how the switch changes."
    )
