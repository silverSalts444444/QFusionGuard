# simulator.py
import numpy as np

class Simulator:
    """
    Variational Quantum Eigensolver (VQE) acting as the environmental simulator.
    Simulates the ground state energy landscape of the plasma confinement Hamiltonian.
    """
    def __init__(self, num_qubits=4):
        self.num_qubits = num_qubits
        self.params = np.random.uniform(0, 2 * np.pi, size=num_qubits)

    def _hamiltonian_expectation(self, weights):
        interaction_term = np.sum(np.cos(weights[:-1] - weights[1:]))
        field_term = np.sum(np.sin(weights))
        return -(interaction_term + 0.5 * field_term)

    def step(self, lr=0.05):
        grads = np.zeros_like(self.params)
        eps = 1e-3
        base_cost = self._hamiltonian_expectation(self.params)
        for i in range(len(self.params)):
            p_plus = self.params.copy()
            p_plus[i] += eps
            grads[i] = (self._hamiltonian_expectation(p_plus) - base_cost) / eps
        self.params -= lr * grads
        return self._hamiltonian_expectation(self.params)