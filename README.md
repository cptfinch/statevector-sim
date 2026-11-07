# statevector-sim

A small state-vector quantum circuit simulator in Python and numpy, written to build a clear, intuitive picture of the linear algebra behind qubits and gates. It is a learning project, not a competitor to Qiskit, Cirq or pytket.

## What it does
- Keeps the state of n qubits as 2**n complex amplitudes.
- Applies one- and two-qubit gates (H, X, Y, Z, S, T, rotations, CNOT, CZ, SWAP) by tensor contraction.
- Computes outcome probabilities (Born rule) and samples measurement results.
- 12 tests: gate unitarity, identities (HH = I, HXH = Z), Bell and GHZ states, qubit ordering, norm preservation, error handling.

```python
from statevector_sim import Circuit
state = Circuit(2).h(0).cnot(0, 1).run()    # Bell state
print(state.vector)                          # [0.707, 0, 0, 0.707]
print(state.sample(1000))                    # about {'00': 500, '11': 500}
```

Run the tests: `uv run --with pytest --with numpy python -m pytest`

## The linear algebra, in my words
<!-- TODO (write this yourself, in your own words; this is the part that matters):
1. Why a qubit state is a vector, and what the two amplitudes mean.
2. Why gates are matrices, and why they must be unitary (what unitary preserves).
3. Why two qubits need a tensor (Kronecker) product, and why 2 qubits means 4 amplitudes.
4. Why entanglement (the Bell state) cannot be written as a product of two single-qubit states.
5. Why simulating n qubits costs memory that doubles with every extra qubit.
-->

## How this was built
<!-- TODO (decide and state honestly): written with AI assistance (Claude) and reviewed and understood by me line by line. -->

## Limitations
Dense state vectors only (memory doubles per qubit), no noise model, no circuit optimiser, and no mid-circuit measurement or classical control.

## Licence
MIT.
