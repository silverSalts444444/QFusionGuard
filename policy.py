# policy.py
import numpy as np

class LocalizedVQC:
    """
    Variational Quantum Circuit (VQC) Main Control Algorithm.
    Features localized entanglement blocks and direct hardware noise variance.
    """
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.weights = np.random.uniform(0, 2 * np.pi, size=(3, num_qubits))

    def _localized_ansatz(self, weights, input_state, noise_std=0.035):
        noisy_weights = weights + np.random.normal(0, noise_std, size=weights.shape)
        state = np.sin(input_state + noisy_weights[0])
        
        # Localized entanglement blocks (0,1) and (2,3)
        entangled = np.zeros_like(state)
        for i in range(0, self.num_qubits, 2):
            if i + 1 < self.num_qubits:
                entangled[i] = state[i] * np.cos(state[i+1])
                entangled[i+1] = state[i+1] * np.sin(state[i])
            else:
                entangled[i] = state[i]
                
        output_signals = np.tanh(entangled + noisy_weights[2])
        return output_signals

    def forward(self, input_state, noise_std=0.035):
        return self._localized_ansatz(self.weights, input_state, noise_std)

    def update(self, input_state, target_energy, lr=0.04, noise_std=0.035):
        grads = np.zeros_like(self.weights)
        eps = 1e-3
        def loss_fn(w):
            signals = self._localized_ansatz(w, input_state, noise_std)
            return np.mean((signals - target_energy)**2)
        base_loss = loss_fn(self.weights)
        for idx in np.ndindex(self.weights.shape):
            w_plus = self.weights.copy()
            w_plus[idx] += eps
            grads[idx] = (loss_fn(w_plus) - base_loss) / eps
        self.weights -= lr * grads
        return base_loss