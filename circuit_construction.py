import numpy as np

from copy import deepcopy

from qiskit import QuantumCircuit
from qiskit import ClassicalRegister
from qiskit import QuantumRegister
from qiskit.circuit.library import U3Gate, RYGate
from qiskit import transpile

# FUNCTIONS TO ENCODE TRAINING POINTS 

# pads input with zeros if length of input is some integer k = log2(input)
def pad_input(x):
    padding = int(2**(np.ceil(np.log2(len(x))))-len(x))
    padded_x = np.pad(x,(0,padding),mode='constant')
    return padded_x

# returns gate to encode x with amplitude encoding 
def amplitude_encoding_gate(x):
    x = pad_input(x)
    x = x/np.linalg.norm(x)
    num_encoding_qubits = int(np.ceil(np.log2(len(x))))

    qc = QuantumCircuit(num_encoding_qubits)
    qc.initialize(x, qc.qubits)
    qc = transpile(qc,basis_gates=["u1","u2","u3","cx"],optimization_level=1)
    # return encoding_gate  
    return qc.to_gate()

# returns gate to encode x with rotation encoding
def rotation_encoding_gate(x):
    num_encoding_qubits = len(x)
    qc = QuantumCircuit(num_encoding_qubits)
    for i in range(len(x)):
        qc.ry(x[i]*2,i)
    return qc.to_gate()

# encodes list of training points with specified encoding
def encode_training_points_instr(X_train,encoding):
    # define sub-circuit to encode training points
    num_training_points = len(X_train) 
    num_index_qubits = int(np.ceil((np.log2(num_training_points)))) 
    if encoding=='amplitude':
        num_encoding_qubits = int(np.ceil(np.log2(len(X_train[0]))))
        sub_qr = QuantumRegister(num_index_qubits+num_encoding_qubits)
        sub_qc = QuantumCircuit(sub_qr, name='encode_training_points')
        for i in range(len(X_train)):
            # format converts index (int) to binary number 
            # [::-1] reverses binary number to account for Qiskit's MSB rule
            # mcry = amplitude_encoding_gate(X_train[i]).control(num_ctrl_qubits = num_index_qubits, ctrl_state = format(i,'b').rjust(num_index_qubits,'0')[::-1])
            mcry = amplitude_encoding_gate(X_train[i]).control(num_ctrl_qubits = num_index_qubits, ctrl_state = format(i,'b').rjust(num_index_qubits,'0'))
            sub_qc.append(mcry,sub_qr)
    if encoding=='rotation':
        num_encoding_qubits = int(len(X_train[0]))
        sub_qr = QuantumRegister(num_index_qubits+num_encoding_qubits)
        sub_qc = QuantumCircuit(sub_qr, name='encode_training_points')
        for i in range(len(X_train)):
            # mcry = rotation_encoding_gate(X_train[i]).control(num_ctrl_qubits = num_index_qubits, ctrl_state = format(i,'b').rjust(num_index_qubits,'0')[::-1])
            mcry = rotation_encoding_gate(X_train[i]).control(num_ctrl_qubits = num_index_qubits, ctrl_state = format(i,'b').rjust(num_index_qubits,'0'))
            sub_qc.append(mcry,sub_qr)
    # convert sub-circuit to instruction  
    return sub_qc.to_instruction()

# FUNCTIONS TO ENCODE TRAINING LABELS 

# returns [theta, phi] for U3 gate for encoding 3-D label vectors 
# DOES NOT normalize vector
def to_spher_coords(vector):
    vector = vector/np.linalg.norm(vector)
    x = vector[0]
    y = vector[1]
    z = vector[2]
    theta = np.arctan2(np.sqrt(x**2 + y**2),z)
    phi = np.arctan2(y,x)  
    return np.round(theta,decimals=6),np.round(phi,decimals=6)  

# encodes list of training labels 
def encode_training_labels_instr(num_index_qubits, y_train, label_vectors):
    # define sub-circuit to encode training labels
    sub_qr = QuantumRegister(num_index_qubits+1)
    sub_qc = QuantumCircuit(sub_qr, name='encode_training_labels')
    for i in range(len(y_train)):
        label_vector = label_vectors[y_train[i]]
        theta,phi = to_spher_coords(label_vector)
        # mcu3 = U3Gate(theta,phi,0).control(num_ctrl_qubits=num_index_qubits, ctrl_state=format(i,'b').rjust(num_index_qubits,'0')[::-1])
        mcu3 = U3Gate(theta,phi,0).control(num_ctrl_qubits=num_index_qubits, ctrl_state=format(i,'b').rjust(num_index_qubits,'0'))
        sub_qc.append(mcu3,sub_qr)
        # sub_qc.barrier()
    return sub_qc.to_instruction()

# adds depolarising noise to the label qubit of qc
def add_depolarising_noise(qc,P):
    label_qubit = qc.qubits[-1]
    sub_qr = QuantumRegister(3,'env')
    qc.add_register(sub_qr)
    theta = (1/2)*np.arccos(1-(2*P))
    for i in range(3):
        qc.ry(theta,sub_qr[i])
    qc.cx(sub_qr[0],label_qubit)
    qc.cy(sub_qr[1],label_qubit)
    qc.cz(sub_qr[2],label_qubit)
    
    return qc 

# test input = test point that needs to be classified
# X_train = array of training points 
# encoding = method of encoding, "amplitude" or "rotation"
# label_vectors = the label vectors used to encode the classed 
# simulator = "qasm" or "statevector"
# P = depolarising probability (P=0 corresponds to no depolarising noise, P=1 corresponds to total loss of information)
def construct_fold_circuit(X_train,y_train,label_vectors,encoding="amplitude",simulator="qasm",P=0):
    # the number of training points 
    num_training_points = len(X_train)

    # the number of index qubits needed in the index register to store each unique training point
    num_index_qubits = int(np.ceil((np.log2(num_training_points))))

    # the number of qubits needed to encode a SINGLE training/test point
    if encoding == 'amplitude':
        num_encoding_qubits = int(np.ceil(np.log2(len(X_train[0]))))

    if encoding == 'rotation':
        num_encoding_qubits = len(X_train[0])

    # define the quantum and classical registers
    ancilla_qubit = QuantumRegister(1,'ancilla')

    test_input_qubits = QuantumRegister(num_encoding_qubits,'test_input')

    index_reg = QuantumRegister(num_index_qubits,'index') 

    training_input_qubits = QuantumRegister(num_encoding_qubits,'training_input')

    label_qubit = QuantumRegister(1,'label')
    
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(ancilla_qubit,test_input_qubits,index_reg,training_input_qubits,label_qubit,cr)

    # encode w_m (training point weights)
    weights = np.ones([num_training_points,])/np.sqrt(num_training_points)
    # weights = np.array([1/np.sqrt(6),1/np.sqrt(6),1/np.sqrt(6),0,1/np.sqrt(6),1/np.sqrt(6),1/np.sqrt(6),0])

    qc.append(amplitude_encoding_gate(weights),index_reg)
    qc.barrier()
   
    # encode training points
    if encoding == 'amplitude':
        sub_inst = encode_training_points_instr(X_train,encoding)
    if encoding == 'rotation':
        sub_inst = encode_training_points_instr(X_train,encoding)

    ctrl_qubits = []

    for i in range(num_index_qubits):
        ctrl_qubits.append(index_reg[i])

    qc.append(sub_inst,ctrl_qubits+list(training_input_qubits))
    
    qc.barrier()
    
    # encode training labels
    sub_inst = encode_training_labels_instr(num_index_qubits,y_train,label_vectors)

    ctrl_qubits = []

    for i in range(num_index_qubits):
        ctrl_qubits.append(index_reg[i])

    ctrl_qubits+list(training_input_qubits)

    qc.append(sub_inst,ctrl_qubits+list(label_qubit))
    
    qc.barrier()
        
    # perfon the three different measurements 
    if simulator == 'qasm':
        qc_x = qc.copy()
        qc_y = qc.copy()
        qc_z = qc.copy()
    
        qc_x.h(label_qubit)
        qc_x.cx(ancilla_qubit,label_qubit)
    
        qc_y.sdg(label_qubit)
        qc_y.h(label_qubit)
        qc_y.cx(ancilla_qubit,label_qubit)
    
        qc_z.cx(ancilla_qubit,label_qubit)

        if P!=0:     
            add_depolarising_noise(qc_x,P)
            add_depolarising_noise(qc_y,P)
            add_depolarising_noise(qc_z,P)

        qc_x.measure(label_qubit,0)
        qc_y.measure(label_qubit,0)
        qc_z.measure(label_qubit,0)

        return [qc_x,qc_y,qc_z]
    
    if simulator == 'statevector':
        return qc 