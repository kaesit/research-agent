from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
 
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()
 
sampler = StatevectorSampler()
result = sampler.run([qc], shots=1024).result()
print(result[0].data.meas.get_counts())

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorOptions
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from matplotlib import pyplot as plt
# Uncomment the next line if you want to use a simulator:
# from qiskit_ibm_runtime.fake_provider import FakeBelemV2
 
 
# Create a new circuit with two qubits
qc = QuantumCircuit(4)
 
# Add a Hadamard gate to qubit 0
qc.h(3)
 
# Perform a controlled-X gate on qubit 1, controlled by qubit 0
qc.cx(0, 1, "11")
qc.cx(0, 2, "11")
qc.cx(1, 2, "11")
qc.cx(0, 3, "11")
qc.cx(1, 3, "11")


 
# Return a drawing of the circuit using MatPlotLib ("mpl").
# These guides are written by using Jupyter notebooks, which
# display the output of the last line of each cell.
# If you're running this in a script, use `print(qc.draw())` to
# print a text drawing.
qc.draw("mpl")
plt.show()
from qiskit.circuit.random import random_circuit
 
qc_random = [(random_circuit(6, 6, measure=True, seed=i)) for i in range(10)]
qc_random[0].draw(output="mpl", idle_wires=False)
plt.show()
