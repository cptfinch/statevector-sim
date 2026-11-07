"""Standard gates as plain numpy matrices.

Every gate is a unitary matrix U (U† U = I), so applying it preserves the
length (norm) of the state vector, which is what keeps probabilities summing to 1.
"""
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)           # bit flip
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)          # phase flip
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)   # superposition
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Two-qubit gates act on |control target>, basis order 00, 01, 10, 11.
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def is_unitary(u: np.ndarray, atol: float = 1e-12) -> bool:
    return np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=atol)
