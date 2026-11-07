"""A tiny circuit: a list of (gate, qubits) steps that can be run on a fresh state."""
from __future__ import annotations
import numpy as np
from . import gates as g
from .state import State


class Circuit:
    def __init__(self, n_qubits: int):
        self.n = n_qubits
        self.steps: list[tuple[np.ndarray, tuple[int, ...]]] = []

    def add(self, gate: np.ndarray, *qubits: int) -> "Circuit":
        self.steps.append((gate, qubits))
        return self

    def h(self, q): return self.add(g.H, q)
    def x(self, q): return self.add(g.X, q)
    def z(self, q): return self.add(g.Z, q)
    def cnot(self, control, target): return self.add(g.CNOT, control, target)

    def run(self) -> State:
        state = State(self.n)
        for gate, qubits in self.steps:
            state.apply(gate, *qubits)
        return state
