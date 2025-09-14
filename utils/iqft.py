from qiskit import QuantumCircuit, transpile
from qiskit.visualization import circuit_drawer
from qiskit_aer import StatevectorSimulator, AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit.quantum_info import Kraus, Statevector, DensityMatrix
import numpy as np
import tempfile

class IQFT:
    def __init__(self, n=2, state=None, noise=False, noise_type=None, noise_options=None, measure=False, comparison=False):
        self.n = n
        self.state = state
        self.qc = None
        self.noise = noise
        self.noise_type = noise_type
        self.noise_options = noise_options
        self.resulting_state = None
        self.resulting_probabilities = None
        self.resulting_counts = None
        self.measure = measure
        self.target = None
        self.comparison = comparison
        self.comp_fidelity = None
        self.comp_counts = None

    def build_iqft(self):
        qc = QuantumCircuit(self.n)
        
        self.qc = self.add_iqft_circuit(qc)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            circuit_drawer(qc, output="mpl", filename=tmp.name, style="iqx-standard")
            img_path = tmp.name
        return img_path

    def add_iqft_circuit(self, qc):
        for qubit in range(self.n):
            qc.h(qubit)
        
        curr_num = 1
        for qubit in reversed(range(self.n)):
            qc.p(int(self.state) * np.pi/curr_num, qubit)
            curr_num *= 2
        
        for qubit in range(self.n // 2):
            qc.swap(qubit, self.n - qubit - 1)
            qc.barrier()
            
        def add_iqft_rotations(circuit, n):
            for target in range(n):
                for control in reversed(range(target)):
                    angle = -np.pi / 2 ** (target - control)
                    circuit.cp(angle, control, target)
                circuit.h(target)
            return circuit
                    
        add_iqft_rotations(qc, self.n)
        
        if self.noise:
            qc.save_density_matrix()
        if self.measure:
            qc.measure_all()
        return qc

    def simulate(self):
        noise_model = None
        if self.noise:
            noise_model = NoiseModel()
            if self.noise_type == "Depolarizing":
                error = depolarizing_error(self.noise_options["depolarizing"], 1)
                noise_model.add_all_qubit_quantum_error(error, ['h'])
            elif self.noise_type == "Amplitude Damping":
                error = amplitude_damping_error(self.noise_options["amplitude_damping"])
                noise_model.add_all_qubit_quantum_error(error, ['h'])
            elif self.noise_type == "Phase Damping":
                error = phase_damping_error(self.noise_options["phase_damping"])
                noise_model.add_all_qubit_quantum_error(error, ['h'])
            elif self.noise_type == "Bit flip":
                error = self.noise_options["bit_flip"]
                K0 = np.sqrt(error) * np.eye(2)
                K1 = np.sqrt(1 - error) * np.array([[0, 1], [1, 0]])
                noise_ops = Kraus([K0, K1])
                noise_model.add_all_qubit_quantum_error(noise_ops, ['h'])
            elif self.noise_type == "Phase flip":
                error = self.noise_options["phase_flip"]
                K0 = np.sqrt(error) * np.eye(2)
                K1 = np.sqrt(1 - error) * np.array([[1, 0], [0, -1]])
                noise_ops = Kraus([K0, K1])
                noise_model.add_all_qubit_quantum_error(noise_ops, ['h'])
        if self.comparison:
            ideal = StatevectorSimulator(precision='single')
            ideal_circ = transpile(self.qc, ideal)
            ideal_job = ideal.run(ideal_circ, shots=1024)
            ideal_result = ideal_job.result()
            ideal_state = Statevector(ideal_result.get_statevector(ideal_circ))
            ideal_counts = ideal_result.get_counts()
            noisy = AerSimulator(precision='single', noise_model=noise_model)
            noisy_circ = transpile(self.qc, noisy)
            noisy_job = noisy.run(noisy_circ, shots=1024)
            noisy_result = noisy_job.result()
            noisy_state = DensityMatrix(noisy_result.data(noisy_circ.name)['density_matrix'])
            noisy_counts = noisy_result.get_counts()
            self.comp_counts = (ideal_counts, noisy_counts)
            self.comp_fidelity = (ideal_state, noisy_state)
        else:
            if not self.noise:
                statevector = StatevectorSimulator(precision='single')
                circ = transpile(self.qc, statevector)
                self.target = statevector.target
                job = statevector.run(circ, shots=1024)
                self.resulting_state = Statevector(job.result().get_statevector(circ))
                self.resulting_probabilities = self.resulting_state.probabilities()
                if self.measure:
                    self.resulting_counts = job.result().get_counts()
            else:
                aer = AerSimulator(precision='single', noise_model=noise_model)
                circ = transpile(self.qc, aer)
                self.target = aer.target
                job = aer.run(circ, shots=1024)
                result = job.result()
                density_matrix = result.data(circ.name)['density_matrix']
                self.resulting_state = DensityMatrix(density_matrix)
                self.resulting_probabilities = self.resulting_state.probabilities()
                if self.measure:
                    self.resulting_counts = result.get_counts()

    def get_resulting_state(self):
        return self.resulting_state
    def get_resulting_probabilities(self):
        return self.resulting_probabilities
    def get_resulting_counts(self):
        return self.resulting_counts
    def get_circuit(self):
        return self.qc
    def get_target(self):
        return self.target
    def get_comp_fidelity(self):
        return self.comp_fidelity
    def get_comp_counts(self):
        return self.comp_counts
