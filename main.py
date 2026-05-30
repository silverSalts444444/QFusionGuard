

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
import logging, os, time

import simulator as sim_module
import doom

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("logs/qfusionguard.log"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger("QFusionGuard")

N_QUBITS  = 6
N_LAYERS  = 4
N_OUTPUTS = 4

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="autograd", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

SENSOR_MIN = np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.05])
SENSOR_MAX = np.array([30.0,  1.0, 80.0, 50.0,  1.0,  5.0 ])
CTRL_MIN   = np.array([4.0,  0.01,  8.0,  1.0])
CTRL_MAX   = np.array([12.0, 0.10, 20.0,  2.5])

def normalise_inputs(sensor_vec):
    clipped = np.clip(sensor_vec, SENSOR_MIN, SENSOR_MAX)
    unit    = (clipped - SENSOR_MIN) / (SENSOR_MAX - SENSOR_MIN + 1e-9)
    return (unit * 2 - 1) * np.pi

def decode_outputs(qnn_out):
    raw  = pnp.stack(qnn_out[:N_OUTPUTS])
    unit = (raw + 1.0) / 2.0
    return CTRL_MIN + unit * (CTRL_MAX - CTRL_MIN)

def loss_fn(weights, sim, n_rollout=3):
    total_loss = pnp.array(0.0)
    for _ in range(n_rollout):
        state = sim.get_state()
        sensor = np.array([
            state["temp_n"],
            state["mag_stability"],
            state["rad_loss"],
            state["efficiency"],
            state["instability"],
            state["confinement"],
        ])
        x       = pnp.array(normalise_inputs(sensor), requires_grad=False)
        qnn_out = quantum_circuit(x, weights)

        # Keep as pennylane tensor for loss calculation
        raw     = pnp.stack(list(qnn_out[:N_OUTPUTS]))
        unit    = (raw + 1.0) / 2.0

        # Detach for simulator using qml.math
        raw_val  = qml.math.toarray(raw)
        unit_val = (raw_val + 1.0) / 2.0
        ctrl_np  = CTRL_MIN + unit_val * (CTRL_MAX - CTRL_MIN)

        sim.update_inputs(
            mag_strength=float(ctrl_np[0]),
            fuel_flow=float(ctrl_np[1]),
            plasma_temp=float(ctrl_np[2]),
            plasma_shape=float(ctrl_np[3]),
        )
        temp_n, mag_stab, rad_loss, efficiency, instability, tau_e = sim.simulate()

        # Loss uses the differentiable tensor path (raw/unit) plus simulator scalars
        ctrl_diff = CTRL_MIN + unit * (CTRL_MAX - CTRL_MIN)
        temp_dev  = (ctrl_diff[0] - sim_module.Simulator.TARGET_TEMP) ** 2 * 3.0
        instab_p  = pnp.array(instability, requires_grad=False) * 50.0
        rad_p     = pnp.array(rad_loss,    requires_grad=False) * 0.3
        eff_p     = -pnp.array(efficiency, requires_grad=False) * 20.0
        magstab_p = (1.0 - pnp.array(mag_stab, requires_grad=False)) * 15.0


        total_loss = total_loss + temp_dev + instab_p + rad_p + eff_p + magstab_p
    return total_loss / n_rollout
def train(n_epochs=300, lr=0.04, doom_interval=50):
    log.info("QFusionGuard — PennyLane VQC Training starting")
    weight_shape = qml.StronglyEntanglingLayers.shape(N_LAYERS, N_QUBITS)
    weights = pnp.random.uniform(-np.pi, np.pi, weight_shape, requires_grad=True)
    opt     = qml.AdamOptimizer(stepsize=lr)
    reactor = sim_module.Simulator()
    doom_engine = doom.DoomAlgorithm(reactor)
    history = {"loss": [], "temp": [], "instability": [], "efficiency": []}

    for epoch in range(n_epochs):
        if epoch > 0 and epoch % doom_interval == 0:
            doom_engine.attack(epoch)
        weights, current_loss = opt.step_and_cost(
            lambda w: loss_fn(w, reactor), weights
        )
        state = reactor.get_state()
        history["loss"].append(float(current_loss))
        history["temp"].append(state["temp_n"])
        history["instability"].append(state["instability"])
        history["efficiency"].append(state["efficiency"])
        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:4d} | Loss={current_loss:8.3f} | "
                f"Temp={state['temp_n']:5.2f} keV | "
                f"Q={state['efficiency']:5.2f} | "
                f"Instab={state['instability']:.3f}"
            )

    doom_engine.summary()
    _plot_training(history, n_epochs)
    return weights, reactor

def run_control_loop(weights, reactor, steps=50):
    log.info("Starting closed-loop control")
    for step in range(steps):
        state  = reactor.get_state()
        sensor = np.array([
            state["temp_n"], state["mag_stability"], state["rad_loss"],
            state["efficiency"], state["instability"], state["confinement"],
        ])
        x       = pnp.array(normalise_inputs(sensor), requires_grad=False)
        qnn_out = quantum_circuit(x, weights)
        ctrl    = decode_outputs([float(v) for v in qnn_out])
        reactor.update_inputs(
            mag_strength=ctrl[0], fuel_flow=ctrl[1],
            plasma_temp=ctrl[2],  plasma_shape=ctrl[3],
        )
        reactor.simulate()
        state = reactor.get_state()
        log.info(
            f"Step {step:3d} | "
            f"B={ctrl[0]:.2f}T  Flow={ctrl[1]:.4f}g/s  "
            f"T={ctrl[2]:.1f}keV  shape={ctrl[3]:.2f} | "
            f"ActualT={state['temp_n']:.2f}  Q={state['efficiency']:.2f}  "
            f"Risk={state['instability']:.3f}"
        )

def _plot_training(history, n_epochs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("QFusionGuard — VQC Training", fontsize=14, fontweight="bold")
    epochs = range(n_epochs)
    axes[0,0].plot(epochs, history["loss"], color="crimson")
    axes[0,0].set_title("Total Loss")
    axes[0,1].plot(epochs, history["temp"], color="darkorange")
    axes[0,1].axhline(15, color="gray", linestyle="--", label="Target 15 keV")
    axes[0,1].set_title("Plasma Temperature (keV)"); axes[0,1].legend()
    axes[1,0].plot(epochs, history["instability"], color="purple")
    axes[1,0].set_title("Disruption Risk")
    axes[1,1].plot(epochs, history["efficiency"], color="steelblue")
    axes[1,1].axhline(10, color="gray", linestyle="--", label="Q=10 target")
    axes[1,1].set_title("Q Factor"); axes[1,1].legend()
    plt.tight_layout()
    plt.savefig("logs/training_curves.png", dpi=120)
    log.info("Saved → logs/training_curves.png")
    plt.show()

if __name__ == "__main__":
    trained_weights, reactor = train(n_epochs=300, lr=0.04, doom_interval=50)
    run_control_loop(trained_weights, reactor, steps=50)