
import numpy as np


class Simulator:
    TARGET_TEMP        = 15.0
    TARGET_EFFICIENCY  = 10.0
    TARGET_INSTABILITY = 0.0
    TARGET_MAG_STABILITY = 1.0

    def __init__(self, mag_strength=7.0, fuel_flow=0.042, plasma_temp=15.0,
                 plasma_shape=1.7, power_in=50.0, coolant_flow=200.0):
        self.mag_strength  = mag_strength
        self.fuel_flow     = fuel_flow
        self.plasma_temp   = plasma_temp
        self.plasma_shape  = plasma_shape
        self.power_in      = power_in
        self.coolant_flow  = coolant_flow

        self._state = {
            "temp_n":        plasma_temp,
            "mag_stability": 0.85,
            "rad_loss":      12.0,
            "efficiency":    1.5,
            "instability":   0.15,
            "confinement":   0.3,
        }

    def simulate(self):
        s  = self._state
        B  = self.mag_strength
        tf = self.fuel_flow
        Tp = self.plasma_temp
        k  = self.plasma_shape
        Pi = self.power_in

        temp_n = (s["temp_n"] * 0.6 + Tp * 0.4
                  + B * 0.1
                  - s["instability"] * 3.0
                  + np.random.normal(0, 0.3))

        mag_stability = np.clip(
            s["mag_stability"] * 0.7
            + (B / 12.0) * 0.25
            - abs(k - 1.7) * 0.05
            + np.random.normal(0, 0.02),
            0.0, 1.0
        )

        tau_e = np.clip(
            0.05 * (B ** 0.8) * (Pi ** -0.69) * k * 10
            + np.random.normal(0, 0.01),
            0.05, 5.0
        )

        rad_loss = np.clip(
            (s["rad_loss"] * 0.5
             + temp_n * B * 0.05
             + (1.0 / (tf + 0.001)) * 0.3
             + np.random.normal(0, 0.5)),
            0.5, 80.0
        )

        sigma_v   = np.exp(-0.1 * (temp_n - 15.0) ** 2)
        power_out = np.clip(
            tf * 1000 * sigma_v * mag_stability * k * 10.0
            + np.random.normal(0, 2.0),
            0.0, 1000.0
        )
        efficiency = power_out / (Pi + 1e-6)

        instability = np.clip(
            s["instability"] * 0.5
            + abs(temp_n - self.TARGET_TEMP) * 0.02
            + (1.0 - mag_stability) * 0.15
            + abs(k - 1.7) * 0.05
            + (rad_loss / 80.0) * 0.1
            + np.random.normal(0, 0.02),
            0.0, 1.0
        )

        self._state = dict(
            temp_n=temp_n,
            mag_stability=mag_stability,
            rad_loss=rad_loss,
            efficiency=efficiency,
            instability=instability,
            confinement=tau_e,
        )
        return (temp_n, mag_stability, rad_loss, efficiency, instability, tau_e)

    def update_inputs(self, mag_strength=None, fuel_flow=None, plasma_temp=None,
                      plasma_shape=None, power_in=None, coolant_flow=None):
        if mag_strength  is not None: self.mag_strength  = float(mag_strength)
        if fuel_flow     is not None: self.fuel_flow     = float(fuel_flow)
        if plasma_temp   is not None: self.plasma_temp   = float(plasma_temp)
        if plasma_shape  is not None: self.plasma_shape  = float(plasma_shape)
        if power_in      is not None: self.power_in      = float(power_in)
        if coolant_flow  is not None: self.coolant_flow  = float(coolant_flow)

    def get_state(self):
        return dict(self._state)

    def inject_disruption(self, kind="plasma_loss"):
        if kind == "plasma_loss":
            self._state["instability"] = min(1.0, self._state["instability"] + 0.6)
            self._state["temp_n"]     *= 0.4
        elif kind == "magnet_quench":
            self._state["mag_stability"] = 0.05
            self._state["instability"]   = min(1.0, self._state["instability"] + 0.5)
        elif kind == "tritium_leak":
            self.fuel_flow = max(0.001, self.fuel_flow * 0.1)
            self._state["instability"]  = min(1.0, self._state["instability"] + 0.3)
        elif kind == "overpower":
            self.power_in = min(120.0, self.power_in * 2.5)
            self._state["rad_loss"]    *= 2.0
            self._state["instability"]  = min(1.0, self._state["instability"] + 0.4)
        elif kind == "sabotage":
            self._state["instability"]   = 1.0
            self._state["mag_stability"] = 0.01
            self._state["temp_n"]       *= 0.2
            self.mag_strength = max(4.0, self.mag_strength * 0.3)