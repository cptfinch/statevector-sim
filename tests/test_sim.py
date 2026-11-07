import numpy as np
import pytest
from statevector_sim import Circuit, State, gates as g

SQ = 1 / np.sqrt(2)


def test_all_gates_are_unitary():
    for u in (g.I, g.X, g.Y, g.Z, g.H, g.S, g.T, g.CNOT, g.CZ, g.SWAP, g.rx(0.7), g.rz(1.3)):
        assert g.is_unitary(u)


def test_initial_state_is_all_zeros_ket():
    assert np.allclose(State(3).vector, [1, 0, 0, 0, 0, 0, 0, 0])


def test_hadamard_twice_is_identity():
    assert np.allclose(g.H @ g.H, g.I)


def test_hxh_equals_z():
    assert np.allclose(g.H @ g.X @ g.H, g.Z)


def test_hadamard_makes_equal_superposition():
    s = Circuit(1).h(0).run()
    assert np.allclose(s.vector, [SQ, SQ])
    assert np.allclose(s.probabilities(), [0.5, 0.5])


def test_x_flips_the_correct_qubit_ordering():
    # qubit 0 is the leftmost bit: X on qubit 0 of 2 gives |10>, index 2
    s = Circuit(2).x(0).run()
    assert np.allclose(s.vector, [0, 0, 1, 0])
    s = Circuit(2).x(1).run()
    assert np.allclose(s.vector, [0, 1, 0, 0])


def test_bell_state():
    s = Circuit(2).h(0).cnot(0, 1).run()
    assert np.allclose(s.vector, [SQ, 0, 0, SQ])
    counts = s.sample(2000, np.random.default_rng(1))
    assert set(counts) <= {"00", "11"}            # the two qubits always agree
    assert abs(counts["00"] - 1000) < 150


def test_ghz_state_three_qubits():
    s = Circuit(3).h(0).cnot(0, 1).cnot(1, 2).run()
    expected = np.zeros(8, dtype=complex); expected[0] = expected[7] = SQ
    assert np.allclose(s.vector, expected)


def test_cnot_with_control_and_target_swapped_order():
    # control qubit 1, target qubit 0, on |01>: control is 1, so target flips -> |11>
    s = Circuit(2).x(1).cnot(1, 0).run()
    assert np.allclose(s.vector, [0, 0, 0, 1])


def test_norm_is_preserved_by_random_circuit():
    rng = np.random.default_rng(7)
    c = Circuit(4)
    for _ in range(40):
        c.add(rng.choice([g.H, g.X, g.Z, g.S, g.T]), int(rng.integers(4)))
        a, b = rng.choice(4, size=2, replace=False)
        c.cnot(int(a), int(b))
    assert np.isclose(np.linalg.norm(c.run().vector), 1.0)


def test_swap_gate_matches_three_cnots():
    s1 = Circuit(2).x(0).run().apply(g.SWAP, 0, 1)
    s2 = Circuit(2).x(0).cnot(0, 1).cnot(1, 0).cnot(0, 1).run()
    assert np.allclose(s1.vector, s2.vector)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        State(2).apply(g.H, 5)
    with pytest.raises(ValueError):
        State(2).apply(g.CNOT, 0)
    with pytest.raises(ValueError):
        State(2).apply(g.CNOT, 0, 0)
