# Quantum Algorithm Simulator

A small Tkinter-based desktop app to build and simulate quantum circuits for the Quantum Fourier Transform (QFT), Inverse QFT (IQFT), and a simple Phase Estimation example using Qiskit and Qiskit-Aer backends.

## Features

- Build QFT/IQFT/Phase Estimation circuits for up to 10 qubits.
- Specify initial basis states for each qubit when using QFT.
- Enable simple noise models (bit-flip, phase-flip, depolarizing, amplitude/phase damping) and set a single parameter per model.
- Run simulations on statevector or noisy Aer simulators.
- View and save circuit visualizations (PNG).
- Basic analysis windows: Bloch spheres, probability distribution, state-city, and count comparison.

## Requirements

- Python 3.10+ recommended
- qiskit, qiskit-aer, qiskit-terra
- Pillow
- numpy
- matplotlib

Install dependencies in a virtual environment using the included `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer to install manually, install the essentials:

```bash
pip install qiskit qiskit-aer pillow numpy matplotlib
```

## Usage

1. Open a terminal in the repository root (`/Users/ninjacornix/dipl/kod`).
2. Run the simulator GUI:

```bash
python simulator.py
```

3. Select an algorithm from the top control bar:
   - QFT: choose the number of qubits and set each qubit's initial basis state.
   - IQFT: enter a single integer (used as a parameter for the IQFT helper in this project).
   - Phase Estimation: uses a simplified phase estimation example with control qubits.

4. Optionally enable noise and open Noise Settings to set the noise parameter.
5. Build the circuit and run the simulation.
6. Use the Analysis menu to open visualizations (available after a successful simulation).

## Notes

- The app is intended for experimentation and teaching. It uses Qiskit simulators locally (no IBM Quantum account required).
- Large numbers of qubits or deep circuits may be slow or memory intensive when using noisy simulation.

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
