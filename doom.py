

import numpy as np
import logging
import random

log = logging.getLogger("QFusionGuard.Doom")

FAILURE_MODES = {
    "plasma_loss":      "Plasma loses confinement, strikes the walls",
    "magnet_quench":    "Superconducting magnet failure -> plasma loss",
    "tritium_leak":     "Tritium leak — fuel depleted, contamination risk",
    "overheating":      "Reactor overheating -> potential explosion risk",
    "power_error":      "Power supply error -> cascading failure chain",
    "hydrogen_explode": "Hydrogen accumulation -> explosion risk",
    "radiation_spike":  "Uncontrolled radiation spike",
    "structure_stress": "Structural stress — wall erosion from plasma",
    "sabotage":         "Full compound sabotage — worst-case scenario",
}

INTERNAL_FAILURES = [
    "plasma_loss", "magnet_quench", "tritium_leak",
    "overheating", "power_error", "hydrogen_explode",
    "radiation_spike", "structure_stress",
]

class DoomAlgorithm:
    def __init__(self, simulator, sabotage_every=3):
        self.sim            = simulator
        self.sabotage_every = sabotage_every
        self._call_count    = 0
        self._history       = []

    def attack(self, epoch):
        self._call_count += 1
        if self._call_count % self.sabotage_every == 0:
            event = "sabotage"
            log.warning(f"[DOOM] EPOCH {epoch} — SABOTAGE: {FAILURE_MODES[event]}")
            self._apply(event)
            self._history.append({"epoch": epoch, "mode": "sabotage", "events": [event]})
        else:
            n_events = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            chosen   = random.sample(INTERNAL_FAILURES, k=n_events)
            log.warning(
                f"[DOOM] EPOCH {epoch} — BREAKDOWN: "
                + " | ".join(FAILURE_MODES[e] for e in chosen)
            )
            for event in chosen:
                self._apply(event)
            self._history.append({"epoch": epoch, "mode": "internal", "events": chosen})

    def _apply(self, event):
        mapping = {
            "plasma_loss":      "plasma_loss",
            "magnet_quench":    "magnet_quench",
            "tritium_leak":     "tritium_leak",
            "overheating":      "overpower",
            "power_error":      "overpower",
            "hydrogen_explode": "plasma_loss",
            "radiation_spike":  "overpower",
            "structure_stress": "plasma_loss",
            "sabotage":         "sabotage",
        }
        self.sim.inject_disruption(kind=mapping.get(event, "plasma_loss"))

    def summary(self):
        n_internal = sum(1 for h in self._history if h["mode"] == "internal")
        n_sabotage = sum(1 for h in self._history if h["mode"] == "sabotage")
        log.info(f"[DOOM] {len(self._history)} attacks: {n_internal} breakdowns, {n_sabotage} sabotages")