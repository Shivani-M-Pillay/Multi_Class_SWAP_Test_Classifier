from qiskit import transpile
from qiskit import Aer
from qiskit import BasicAer
from qiskit import execute
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector

from circuit_construction import *

RUNS = 10
SHOTS = 8192

# FUNCTIONS TO EXECUTE AND INTERPRET CIRCUIT FOR STATEVECTOR SIMULATOR 

# executes each circuit using statevector simulator
def execute_circuit_for_sv_sim(sv,qc):
    sv = sv.evolve(qc)
    return sv

# extracts predicted vector from the statevector
def extract_pred_vector_sv(statevector,P,test_point_num):
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j,0]])
    Z = np.array([[1,0],[0,-1]])
     
    if P!=0:
        # num_system_qubits = int(np.log2(statevectors[0].shape[0]))-3
        num_system_qubits = int(np.log2(statevector.dim)) # Statevector object has its own attribute for its length 

        qc = QuantumCircuit(num_system_qubits)

        qc_x = qc.copy()
        qc_y = qc.copy()
        qc_z = qc.copy()
    
        qc_x.h(num_system_qubits-1)
        qc_x.cx(0,num_system_qubits-1)
    
        qc_y.sdg(num_system_qubits-1)
        qc_y.h(num_system_qubits-1)
        qc_y.cx(0,num_system_qubits-1)
    
        qc_z.cx(0,num_system_qubits-1)

        add_depolarising_noise(qc_x,P)
        add_depolarising_noise(qc_y,P)
        add_depolarising_noise(qc_z,P)

        statevector = statevector.expand(Statevector.from_label('000'))
        statevectors = [execute_circuit_for_sv_sim(statevector,qc_x),execute_circuit_for_sv_sim(statevector,qc_y),execute_circuit_for_sv_sim(statevector,qc_z)]

        statevectors_data = []
        for i in range(len(statevectors)):
            statevectors_data.append(statevectors[i].data)

        np.save("statevectors_for_test_point_"+str(test_point_num)+".npy",np.array(statevectors_data))

        rho_x = np.outer(statevectors[0],np.transpose(np.conjugate(statevectors[0])))
        rho_y = np.outer(statevectors[1],np.transpose(np.conjugate(statevectors[1])))
        rho_z = np.outer(statevectors[2],np.transpose(np.conjugate(statevectors[2])))

        obs = np.kron(np.eye(2**(3)),np.kron(Z,np.eye(2**(num_system_qubits-1))))
        x = np.trace(np.dot(obs,rho_x))
        y = np.trace(np.dot(obs,rho_y))
        z = np.trace(np.dot(obs,rho_z))

    else:

        # append change of basis and cx to each circuit
        num_system_qubits = int(np.log2(statevector.dim))

        qc = QuantumCircuit(num_system_qubits)

        qc_x = qc.copy()
        qc_y = qc.copy()
        qc_z = qc.copy()
    
        qc_x.h(num_system_qubits-1)
        qc_x.cx(0,num_system_qubits-1)
    
        qc_y.sdg(num_system_qubits-1)
        qc_y.h(num_system_qubits-1)
        qc_y.cx(0,num_system_qubits-1)
    
        qc_z.cx(0,num_system_qubits-1)

        statevectors = [execute_circuit_for_sv_sim(statevector,qc_x),execute_circuit_for_sv_sim(statevector,qc_y),execute_circuit_for_sv_sim(statevector,qc_z)]

        statevectors_data = []
        for i in range(len(statevectors)):
            statevectors_data.append(statevectors[i].data)
        
        np.save("statevectors_for_test_point_"+str(test_point_num)+".npy",np.array(statevectors_data))

        obs = np.kron(Z,np.eye(2**(num_system_qubits-1)))
        rho_x = np.outer(statevectors[0],np.transpose(np.conjugate(statevectors[0])))
        rho_y = np.outer(statevectors[1],np.transpose(np.conjugate(statevectors[1])))
        rho_z = np.outer(statevectors[2],np.transpose(np.conjugate(statevectors[2])))

        # for 'analytic' depolarising noise without simulating it in circuit
        # follows formula from Nielsen and Chuang 
        # rho_x = (P/2)*np.eye(2**num_system_qubits) + (1-P)*rho_x
        # rho_y = (P/2)*np.eye(2**num_system_qubits) + (1-P)*rho_y
        # rho_z = (P/2)*np.eye(2**num_system_qubits) + (1-P)*rho_z

        x = np.trace(np.dot(obs,rho_x))
        y = np.trace(np.dot(obs,rho_y))
        z = np.trace(np.dot(obs,rho_z))

    pred_vector = [float(x),float(y),float(z)]

    return pred_vector

# classifies a given test point using the statevector simulator to execute the circuits
def statevector_classification(X_test, y_test, training_sv, num_index_qubits, y_train, label_vectors, encoding, test_point_num, P=0):
    classification = {}
    # qc = construct_fold_circuit(X_train,y_train,label_vectors,encoding,simulator="statevector",P=P)
    num_system_qubits = int(np.log2(training_sv.dim))
    qc = QuantumCircuit(num_system_qubits)

    # the number of qubits needed to encode a SINGLE training/test point
    if encoding == 'amplitude':
        num_encoding_qubits = int(np.ceil(np.log2(len(X_test))))
        qc.append(amplitude_encoding_gate(X_test),range(1,num_encoding_qubits+1))
    if encoding == 'rotation':
        num_encoding_qubits = len(X_test)
        qc.append(rotation_encoding_gate(X_test),range(1,num_encoding_qubits+1))

    # perform SWAP test
    qc.h(0)
    
    for i in range(num_encoding_qubits):
        qc.cswap(0,1+i,1+num_encoding_qubits+i+num_index_qubits)    
    
    qc.h(0)
    qc.barrier()

    statevector = execute_circuit_for_sv_sim(training_sv, qc)
    pred_vector = extract_pred_vector_sv(statevector,P,test_point_num)
    pred_class = max(np.unique(y_train),key = lambda x : np.dot(pred_vector,label_vectors[x]))

    classification['Test Input'] = X_test.tolist()
    print("Predicted Vector:",pred_vector)
    print("Predicted Class:",pred_class)
    print("True Class:",y_test)

    
    classification["Predicted Vector"] = pred_vector

    classification["True Class"] = int(y_test)

    classification["Predicted Class"] = int(pred_class)

    return pred_class, classification


"""
# FUNCTIONS TO EXECUTE AND INTERPRET CIRCUIT FOR QASM SIMULATOR 

# executes the 3 circuits using the QASM simulator
# def execute_circuits_for_qasm_sim(circuits):
#     backend = Aer.get_backend('qasm_simulator')    
#     # stores a single predicted vector from one RUN
#     pred_vector = np.zeros([3,])
#     # stores each predicted vector in a row (one ROW for each RUN)
#     pred_vector_arr = np.zeros([RUNS,3])
#     for j in range(RUNS):
#         for i in range(3):
#             tcirc = transpile(circuits[i],backend)
#             jobs = execute(tcirc, backend=backend, shots=SHOTS)
#             counts = jobs.result().get_counts()
#             # pred_vector[i] = (counts['0'.rjust(NUM_INDEX_QUBITS+4,'0')]-counts['1'.rjust(NUM_INDEX_QUBITS+4,'0')])/SHOTS
#             pred_vector[i] = (counts['0']-counts['1'])/SHOTS
#         pred_vector_arr[j] = pred_vector
#     return pred_vector_arr

def execute_circuits_for_qasm_sim(circuits):
    simulator = AerSimulator(method='matrix_product_state')  
    # stores a single predicted vector from one RUN
    pred_vector = np.zeros([3,])
    # stores each predicted vector in a row (one ROW for each RUN)
    pred_vector_arr = np.zeros([RUNS,3])
    for j in range(RUNS):
        for i in range(3):
            tcirc = transpile(circuits[i],simulator)
            jobs = execute(tcirc, backend=simulator, shots=SHOTS)
            counts = jobs.result().get_counts()
            # pred_vector[i] = (counts['0'.rjust(NUM_INDEX_QUBITS+4,'0')]-counts['1'.rjust(NUM_INDEX_QUBITS+4,'0')])/SHOTS
            pred_vector[i] = (counts['0']-counts['1'])/SHOTS
        pred_vector_arr[j] = pred_vector
    return pred_vector_arr

# classifies a given test point using the QASM simulator to execute the circuits 
def qasm_classification(test_input,X_train,y_train,y_test,label_vectors,encoding,P=0):
    circuits = construct_circuits(test_input,X_train,y_train,label_vectors,encoding,simulator="qasm",P=P)
    pred_vector = np.zeros([3,])
    pred_vector_arr = execute_circuits_for_qasm_sim(circuits)
    print(pred_vector_arr)
    mean_pred_vector = np.mean(pred_vector_arr,axis=0)
    std_pred_vector = np.std(pred_vector_arr,axis=0)
    pred_class = max(np.unique(y_train),key = lambda x : np.dot(mean_pred_vector,label_vectors[x]))
    # print mean pred_vector and it's deviations 
    print("Predicted State:",end='[')
    print(mean_pred_vector[0],"+-",std_pred_vector[0],end=',  ')
    print(mean_pred_vector[1],"+-",std_pred_vector[1],end=',  ')
    print(mean_pred_vector[2],"+-",std_pred_vector[2],']')
    print("Predicted Class:",pred_class)
    return pred_class
"""