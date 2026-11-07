"""The state of n qubits as a vector of 2**n complex amplitudes.

Convention: qubit 0 is the leftmost bit of a basis label, like the textbook
ket |q0 q1 q2>. So for 2 qubits the amplitudes are ordered |00>, |01>, |10>, |11>.

Internally the vector is kept as a tensor of shape (2,)*n, one axis per qubit,
so a gate on qubit k is a contraction along axis k.
"""
from __future__ import annotations
import numpy as np


class State:
    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("need at least one qubit")
        self.n = n_qubits
        self.tensor = np.zeros((2,) * n_qubits, dtype=complex)
        self.tensor[(0,) * n_qubits] = 1.0           # start in |00...0>

    @property
    def vector(self) -> np.ndarray:
        return self.tensor.reshape(-1)

    def apply(self, gate: np.ndarray, *qubits: int) -> "State":
        """Apply a 1- or 2-qubit gate to the given qubit(s). Returns self."""
        k = len(qubits)
        if gate.shape != (2 ** k, 2 ** k):
            raise ValueError(f"gate shape {gate.shape} does not fit {k} qubit(s)")
        if len(set(qubits)) != k or any(q < 0 or q >= self.n for q in qubits):
            raise ValueError(f"bad qubit indices {qubits} for {self.n} qubits")
        g = gate.reshape((2,) * (2 * k))              # (out..., in...)
        in_axes = list(range(k, 2 * k))
        t = np.tensordot(g, self.tensor, axes=(in_axes, list(qubits)))
        self.tensor = np.moveaxis(t, list(range(k)), list(qubits))
        return self

    def probabilities(self) -> np.ndarray:
        """Born rule: probability of each basis state is |amplitude|**2."""
        return np.abs(self.vector) ** 2

    def sample(self, shots: int, rng: np.random.Generator | None = None) -> dict[str, int]:
        rng = rng or np.random.default_rng()
        outcomes = rng.choice(2 ** self.n, size=shots, p=self.probabilities())
        counts: dict[str, int] = {}
        for o in outcomes:
            label = format(int(o), f"0{self.n}b")
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))
