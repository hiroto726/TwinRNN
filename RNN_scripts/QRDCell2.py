# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:53:11 2024

@author: Hiroto Imamura
"""



import tensorflow as tf
from tensorflow.keras.layers import Layer

class QRDcell2(Layer):
    #made the shape compatible with normal tensorflow operation
    #(batch, 2*unit, 2*unit)=(batch_num, input_dim, output_dim)
    def __init__(self, start=0,nUnit=0, **kwargs):
        super(QRDcell2, self).__init__(**kwargs)
        self.start = start
        self.nUnit=nUnit



    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        # Initial state for Q: identity matrix
        # Initial state for Q: identity matrix
        #batch_size = self.batch_size if batch_size is None else batch_size
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        initial_Q = tf.linalg.eye(self.nUnit*2, batch_shape=[batch_size], dtype=dtype)
        # Initial state for iteration: zeros
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [initial_Q, initial_iteration]


    @property
    def state_size(self):
        return [[self.nUnit*2, self.nUnit*2], 1]
    @tf.function
    def call(self, inputs, states):
        # inputs is J_{n} (batch, nUnit, nUnit)
        # states is Q_{n-1} (batch, nUnit, nUnit). Q_{0} is identity matrix
        # Q_{n}R_{n} = J_{n}Q_{n-1}
        # Q_{n} is the next state. (Q_new)
        # R_{n} is the output. (R_new)
        
        J = inputs#(batch, 2*unit, 2*unit)=(batch_num, input_dim, output_dim)
        Q = states[0]#(batch, 2*unit, 2*unit)
        iteration = states[1]
        # Debug prints
        if iteration[0] >= self.start:
            Q_new, R_new = tf.linalg.qr(tf.matmul(J, Q))#(batch, 2*unit, 2*unit)
        else:
            Q_new = Q
            R_new = tf.zeros_like(Q)
        Rdiag=tf.linalg.diag_part(R_new)#(batch, 2*unit)
        #Rdiag=Rdiag[:,:10]#(batch,10)
        
        iteration += 1
        return Rdiag, [Q_new, iteration]









import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import AbstractRNNCell

class QRDcell2_flat(AbstractRNNCell):
    #made the shape compatible with normal tensorflow operation
    #(batch, 2*unit, 2*unit)=(batch_num, input_dim, output_dim)
    def __init__(self, start=0,nUnit=0, **kwargs):
        super(QRDcell2_flat, self).__init__(**kwargs)
        self.start = start
        self.nUnit=nUnit



    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Initializes the states:
        - Q: Identity matrix of shape (batch_size, 2*nUnit, 2*nUnit)
        - iteration: Tensor of ones with shape (batch_size, 1)
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        
        initial_Q = tf.linalg.eye(self.nUnit*2, batch_shape=[batch_size], dtype=dtype)
        #flatten Q
        Q_flat = tf.reshape(initial_Q, [tf.shape(initial_Q)[0], -1])  # (batch, 4*units^2)
        
        # Initial state for iteration: zeros
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [Q_flat, initial_iteration]


    @property
    def state_size(self):
        return [self.nUnit*2*self.nUnit*2, 1]
    @tf.function
    def call(self, inputs, states):
        """
        Executes one timestep of the QRD cell.
        
        Args:
            inputs: Tensor of shape (batch, 2*nUnit, 2*nUnit)
            states: List containing [Q, iteration]
        
        Returns:
            Rdiag: Tensor of shape (batch, 2*nUnit)
            new_states: [Q_new, iteration_new]
        """
        
        Jacobian_flat = inputs# [batch_size, 4*nUnit^2]-> maybe [batch_size,timesteps, 4*nUnit^2]-
        J = tf.reshape(Jacobian_flat, [tf.shape(Jacobian_flat)[0], self.nUnit*2, self.nUnit*2])#(batch, 2*unit, 2*unit)=(batch_num, input_dim, output_dim)
        Q_flat = states[0]#(batch, 2*unit, 2*unit)
        Q = tf.reshape(Q_flat, [tf.shape(Q_flat)[0], self.nUnit*2, self.nUnit*2])#unflatten Q
        
        iteration = states[1]
        # Debug prints

        if iteration[0] >= self.start:
            Q_new, R_new = tf.linalg.qr(tf.matmul(J, Q))#(batch, 2*unit, 2*unit)
        else:
            Q_new = Q
            R_new = tf.zeros_like(Q)
        Rdiag=tf.linalg.diag_part(R_new)#(batch, 2*unit)

        
        #Rdiag=Rdiag[:,:10]#(batch,10)
        
        iteration_new = iteration + 1
        
        
        #flatten Q
        Q_flat = tf.reshape(Q_new, [tf.shape(Q_new)[0], -1])  # (batch, 4*units^2)
        return Rdiag, [Q_flat, iteration_new]


class QRDcell2_flat_2(AbstractRNNCell):
    """
    A QR decomposition cell that accepts a flattened Jacobian (of shape [batch, 4*nUnit^2]),
    reshapes it to a square matrix (batch, 2*nUnit, 2*nUnit), and then uses QR 
    decomposition to compute the local Lyapunov exponents.
    """
    def __init__(self, start=0, nUnit=0, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Initializes the states:
         - Q: Identity matrix of shape (batch_size, 2*nUnit, 2*nUnit) flattened to (batch_size, 4*nUnit^2)
         - iteration: Tensor of zeros of shape (batch_size, 1) (iteration counter)
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        
        initial_Q = tf.linalg.eye(2 * self.nUnit, batch_shape=[batch_size], dtype=dtype)
        # Flatten Q to a vector for each batch element.
        Q_flat = tf.reshape(initial_Q, [tf.shape(initial_Q)[0], -1])
        
        # Initialize iteration counter to zeros.
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [Q_flat, initial_iteration]

    @property
    def state_size(self):
        # The first state is Q flattened: (2*nUnit)^2, and the second state is the iteration counter.
        return [(2 * self.nUnit) ** 2, 1]

    @tf.function
    def compute_qr(self, J, Q):
        prod = tf.matmul(J, Q)
        # You can also use full_matrices=False to reduce memory footprint:
        Q_new, R_new = tf.linalg.qr(prod, full_matrices=False)
        return Q_new, R_new


    @tf.function
    def call(self, inputs, states):
        """
        Executes one timestep of the QRD cell.
        
        Args:
            inputs: A flattened Jacobian tensor of shape (batch, 4*nUnit^2).
            states: A list containing [Q_flat, iteration].
            
        Returns:
            Rdiag: The diagonal of R (from QR decomposition) as a tensor of shape (batch, 2*nUnit).
            new_states: The updated state as a list [Q_flat, iteration_new].
        """
        # Reshape the flattened Jacobian back to a square matrix.
        Jacobian_flat = inputs
        J = tf.reshape(Jacobian_flat, [tf.shape(Jacobian_flat)[0], 2 * self.nUnit, 2 * self.nUnit])
        
        # Unflatten Q from the states.
        Q_flat = states[0]
        Q = tf.reshape(Q_flat, [tf.shape(Q_flat)[0], 2 * self.nUnit, 2 * self.nUnit])
        
        iteration = states[1]  # Shape: (batch, 1)
        
        # Using tf.cond to decide whether to update Q based on iteration counter.
        # We assume that all batch members follow the same iteration count, so we use the first element.
        def update_Q():
            # Multiply Jacobian and Q to propagate the tangent vectors.
            Q_new, R_new = self.compute_qr(J, Q)
            return Q_new, R_new

        def no_update():
            # Before reaching the start iteration, pass Q unchanged and use a zero R.
            return Q, tf.zeros_like(Q)

        # Get a scalar from the iteration tensor.
        iter_scalar = tf.cast(iteration[0, 0], tf.int32)
        Q_new, R_new = tf.cond(tf.greater_equal(iter_scalar, self.start),
                                 update_Q,
                                 no_update)
        
        # Extract diagonal of R.
        Rdiag = tf.linalg.diag_part(R_new)  # Shape: (batch, 2*nUnit)
        
        # Increase the iteration counter.
        iteration_new = iteration + 1
        
        # Flatten the new Q for the state.
        Q_flat_new = tf.reshape(Q_new, [tf.shape(Q_new)[0], -1])
        
        return Rdiag, [Q_flat_new, iteration_new]


class QRDcell2_flat_half(AbstractRNNCell):
    """
    A QR decomposition cell that accepts a flattened Jacobian (of shape [batch, 4*nUnit^2]),
    reshapes it to a square matrix (batch, 2*nUnit, 2*nUnit), and then uses QR 
    decomposition to compute the local Lyapunov exponents.
    """
    def __init__(self, start=0, nUnit=0, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Initializes the states:
         - Q: Identity matrix of shape (batch_size, 2*nUnit, 2*nUnit) flattened to (batch_size, 4*nUnit^2)
         - iteration: Tensor of zeros of shape (batch_size, 1) (iteration counter)
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        
        initial_Q = tf.linalg.eye(self.nUnit, batch_shape=[batch_size], dtype=dtype)
        # Flatten Q to a vector for each batch element.
        Q_flat = tf.reshape(initial_Q, [tf.shape(initial_Q)[0], -1])
        
        # Initialize iteration counter to zeros.
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [Q_flat, initial_iteration]

    @property
    def state_size(self):
        # The first state is Q flattened: (2*nUnit)^2, and the second state is the iteration counter.
        return [(self.nUnit) ** 2, 1]

    @tf.function
    def compute_qr(self, J, Q):
        prod = tf.matmul(J, Q)
        # You can also use full_matrices=False to reduce memory footprint:
        Q_new, R_new = tf.linalg.qr(prod, full_matrices=False)
        return Q_new, R_new


    @tf.function
    def call(self, inputs, states):
        """
        Executes one timestep of the QRD cell.
        
        Args:
            inputs: A flattened Jacobian tensor of shape (batch, 4*nUnit^2).
            states: A list containing [Q_flat, iteration].
            
        Returns:
            Rdiag: The diagonal of R (from QR decomposition) as a tensor of shape (batch, 2*nUnit).
            new_states: The updated state as a list [Q_flat, iteration_new].
        """
        # Reshape the flattened Jacobian back to a square matrix.
        Jacobian_flat = inputs
        J = tf.reshape(Jacobian_flat, [tf.shape(Jacobian_flat)[0], self.nUnit, self.nUnit])
        
        # Unflatten Q from the states.
        Q_flat = states[0]
        Q = tf.reshape(Q_flat, [tf.shape(Q_flat)[0], self.nUnit, self.nUnit])
        
        iteration = states[1]  # Shape: (batch, 1)
        
        # Using tf.cond to decide whether to update Q based on iteration counter.
        # We assume that all batch members follow the same iteration count, so we use the first element.
        def update_Q():
            # Multiply Jacobian and Q to propagate the tangent vectors.
            Q_new, R_new = self.compute_qr(J, Q)
            return Q_new, R_new

        def no_update():
            # Before reaching the start iteration, pass Q unchanged and use a zero R.
            return Q, tf.zeros_like(Q)

        # Get a scalar from the iteration tensor.
        iter_scalar = tf.cast(iteration[0, 0], tf.int32)
        Q_new, R_new = tf.cond(tf.greater_equal(iter_scalar, self.start),
                                 update_Q,
                                 no_update)
        
        # Extract diagonal of R.
        Rdiag = tf.linalg.diag_part(R_new)  # Shape: (batch, 2*nUnit)
        
        # Increase the iteration counter.
        iteration_new = iteration + 1
        
        # Flatten the new Q for the state.
        Q_flat_new = tf.reshape(Q_new, [tf.shape(Q_new)[0], -1])
        
        return Rdiag, [Q_flat_new, iteration_new]


class Get_norm_ratio(AbstractRNNCell):
    """
    This cell propagates a set of fixed test vectors (tangent directions) and computes 
    the norm ratios after applying the Jacobian. We assume the test matrix (test_mat) 
    is provided in shape (2*nUnit, ncol), meaning that its columns are the initial tangent vectors.
    
    The cell’s state consists of:
      - mat: a tensor of shape (batch, 2*nUnit, ncol) containing the current tangent vectors (flattened in state)
      - iteration: a counter (batch, 1)
    """
    def __init__(self, start=0, nUnit=0, test_mat=None, update_state=True, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit
        self.update_state=update_state
        
        # If no test_mat is provided, use a default with one column.
        # The default shape will be (2*nUnit, 1)
        if test_mat is None:
            self.test_mat = tf.ones((2 * self.nUnit, 1), dtype=tf.float32)
        else:
            self.test_mat = tf.convert_to_tensor(test_mat, dtype=tf.float32)
            
        # Normalize each column of test_mat.
        # For a matrix with shape (2*nUnit, ncol), norm over axis 0 yields a row of norms
        norm_test = tf.norm(self.test_mat, axis=0, keepdims=True)
        self.test_mat = tf.math.divide_no_nan(self.test_mat, norm_test)
        
        # Determine the number of columns (ncol)
        self.ncol = int(tf.shape(self.test_mat)[1])

    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Initializes the state:
          - mat: Broadcasted test_mat with shape (batch, 2*nUnit, ncol) and normalized (each column is unit norm)
          - iteration: A counter set to zeros (batch, 1)
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        # Broadcast test_mat of shape (2*nUnit, ncol) to (batch, 2*nUnit, ncol)
        mat = tf.broadcast_to(self.test_mat, (batch_size, 2 * self.nUnit, self.ncol))
        # Normalize each column vector (each column is of length 2*nUnit) along the vector dimension (axis 1)
        mat =tf.math.divide_no_nan(mat, tf.norm(mat, axis=1, keepdims=True))
        
        # Initial iteration counter: zeros of shape (batch, 1)
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.float32)
        # Flatten mat to match state_size: shape (batch, 2*nUnit * ncol)
        mat_flat = tf.reshape(mat, [batch_size, -1])
        mat_flat=tf.cast(mat_flat,dtype=tf.float32)
        return [mat_flat, initial_iteration]
    
    @property
    def state_size(self):
        # The state consists of the flattened mat (shape: 2*nUnit * ncol) and the iteration (shape: 1)
        return [2 * self.nUnit * self.ncol, 1]
    
    @tf.function
    def call(self, inputs, states):
        """
        On each time step:
          - Reshape the state to obtain mat of shape (batch, 2*nUnit, ncol)
          - Reshape the flattened Jacobian (inputs) to shape (batch, 2*nUnit, 2*nUnit)
          - Compute the product: mat_after = tf.matmul(J, mat), obtaining (batch, 2*nUnit, ncol)
          - Compute the norm of each updated column vector (norm over axis=1)
          - Optionally update the state if the iteration counter exceeds a threshold.
          - Return norm_ratio (shape: batch x ncol) and updated state.
        """
        # Unpack states: mat is stored in flattened form and iteration is a counter.
        mat_flat = states[0]
        iteration = states[1]
        batch_size = tf.shape(mat_flat)[0]
        
        # Reshape the flattened mat to (batch, 2*nUnit, ncol)
        mat = tf.reshape(mat_flat, [batch_size, 2 * self.nUnit, self.ncol])
        #mat = tf.broadcast_to(self.test_mat, (batch_size, 2 * self.nUnit, self.ncol))
        
        
        # Reshape the flattened Jacobian to a square matrix (batch, 2*nUnit, 2*nUnit)
        Jacobian_flat = inputs
        J = tf.reshape(Jacobian_flat, [batch_size, 2 * self.nUnit, 2 * self.nUnit])

        # Multiply: propagate the tangent vectors using the Jacobian.
        # With test_mat in shape (2*nUnit, ncol), the product is:
        # tf.matmul(J, mat) -> (batch, 2*nUnit, ncol)
        mat_after = tf.matmul(J, mat)
        
        # Compute the norm of each column vector.
        # The column vector is in the 2*nUnit dimension (axis=1), so compute norm along axis=1.
        mat_after_norm = tf.norm(mat_after, axis=1, keepdims=True)  # shape: (batch, 1, ncol)
        norm_ratio = tf.squeeze(mat_after_norm, axis=1)  # shape: (batch, ncol)
        
        # Decide whether to update mat: use tf.cond since iteration is a tensor.
        # Here we extract a scalar from the iteration (assuming all batch members are synchronized).
        iter_scalar = tf.cast(iteration[0, 0], tf.int32)
        
        condition = tf.greater_equal(iter_scalar, self.start)
        
        # normalize mat
        condition_mat=tf.math.logical_and(condition,self.update_state)
        mat = tf.where(condition_mat,
                       tf.math.divide_no_nan(mat_after, mat_after_norm),  # if condition True, use updated value
                       mat)  # else, retain the original
        
        # And for norm_ratio:
        norm_ratio = tf.where(condition,
                                norm_ratio,                # if condition True, use computed norm_ratio
                                tf.ones_like(norm_ratio))  # else, use ones
        

        iteration_new = iteration + 1
        
        # Flatten mat for the updated state.
        mat_new = tf.reshape(mat, [batch_size, -1])
        return norm_ratio, [mat_new, iteration_new]

        
class Get_lyap_dir(AbstractRNNCell):
    """
    This cell propagates a set of fixed test vectors (tangent directions) and computes 
    the norm ratios after applying the Jacobian. We assume the test matrix (test_mat) 
    is provided in shape (2*nUnit, ncol), meaning that its columns are the initial tangent vectors.
    
    The cell’s state consists of:
      - mat: a tensor of shape (batch, 2*nUnit, ncol) containing the current tangent vectors (flattened in state)
      - iteration: a counter (batch, 1)
    """
    def __init__(self, start=0, nUnit=0, test_mat=None, update_state=True, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit
        self.update_state=update_state
        
        # If no test_mat is provided, use a default with one column.
        # The default shape will be (2*nUnit, 1)
        if test_mat is None:
            self.test_mat = tf.ones((2 * self.nUnit, 1), dtype=tf.float32)
        else:
            self.test_mat = tf.convert_to_tensor(test_mat, dtype=tf.float32)
            
        # Normalize each column of test_mat.
        # For a matrix with shape (2*nUnit, ncol), norm over axis 0 yields a row of norms
        norm_test = tf.norm(self.test_mat, axis=0, keepdims=True)
        self.test_mat = tf.math.divide_no_nan(self.test_mat, norm_test)
        
        # Determine the number of columns (ncol)
        self.ncol = int(tf.shape(self.test_mat)[1])

    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Initializes the state:
          - mat: Broadcasted test_mat with shape (batch, 2*nUnit, ncol) and normalized (each column is unit norm)
          - iteration: A counter set to zeros (batch, 1)
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        # Broadcast test_mat of shape (2*nUnit, ncol) to (batch, 2*nUnit, ncol)
        mat = tf.broadcast_to(self.test_mat, (batch_size, 2 * self.nUnit, self.ncol))
        # Normalize each column vector (each column is of length 2*nUnit) along the vector dimension (axis 1)
        mat =tf.math.divide_no_nan(mat, tf.norm(mat, axis=1, keepdims=True))
        
        # Initial iteration counter: zeros of shape (batch, 1)
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.float32)
        # Flatten mat to match state_size: shape (batch, 2*nUnit * ncol)
        mat_flat = tf.reshape(mat, [batch_size, -1])
        mat_flat=tf.cast(mat_flat,dtype=tf.float32)
        return [mat_flat, initial_iteration]
    
    @property
    def state_size(self):
        # The state consists of the flattened mat (shape: 2*nUnit * ncol) and the iteration (shape: 1)
        return [2 * self.nUnit * self.ncol, 1]
    
    @tf.function
    def call(self, inputs, states):
        """
        On each time step:
          - Reshape the state to obtain mat of shape (batch, 2*nUnit, ncol)
          - Reshape the flattened Jacobian (inputs) to shape (batch, 2*nUnit, 2*nUnit)
          - Compute the product: mat_after = tf.matmul(J, mat), obtaining (batch, 2*nUnit, ncol)
          - Compute the norm of each updated column vector (norm over axis=1)
          - Optionally update the state if the iteration counter exceeds a threshold.
          - Return norm_ratio (shape: batch x ncol) and updated state.
        """
        # Unpack states: mat is stored in flattened form and iteration is a counter.
        mat_flat = states[0]
        iteration = states[1]
        batch_size = tf.shape(mat_flat)[0]
        
        # Reshape the flattened mat to (batch, 2*nUnit, ncol)
        mat = tf.reshape(mat_flat, [batch_size, 2 * self.nUnit, self.ncol])
        #mat = tf.broadcast_to(self.test_mat, (batch_size, 2 * self.nUnit, self.ncol))
        
        
        # Reshape the flattened Jacobian to a square matrix (batch, 2*nUnit, 2*nUnit)
        Jacobian_flat = inputs
        J = tf.reshape(Jacobian_flat, [batch_size, 2 * self.nUnit, 2 * self.nUnit])

        # Multiply: propagate the tangent vectors using the Jacobian.
        # With test_mat in shape (2*nUnit, ncol), the product is:
        # tf.matmul(J, mat) -> (batch, 2*nUnit, ncol)
        mat_after = tf.matmul(J, mat)
        
        mat_project=tf.reduce_sum(mat*mat_after,axis=1)#-> (batch, ncol)
        
        # Compute the norm of each column vector.
        # The column vector is in the 2*nUnit dimension (axis=1), so compute norm along axis=1.
        mat_after_norm = tf.norm(mat_after, axis=1, keepdims=True)  # shape: (batch, 1, ncol)
        #norm_ratio = tf.squeeze(mat_after_norm, axis=1)  # shape: (batch, ncol)
        
        
        
        # Decide whether to update mat: use tf.cond since iteration is a tensor.
        # Here we extract a scalar from the iteration (assuming all batch members are synchronized).
        iter_scalar = tf.cast(iteration[0, 0], tf.int32)
        
        condition = tf.greater_equal(iter_scalar, self.start)
        
        # normalize mat
        condition_mat=tf.math.logical_and(condition,self.update_state)
        mat = tf.where(condition_mat,
                       tf.math.divide_no_nan(mat_after, mat_after_norm),  # if condition True, use updated value
                       mat)  # else, retain the original
        
        # And for norm_ratio:
        norm_ratio = tf.where(condition,
                                mat_project,                # if condition True, use computed norm_ratio
                                tf.ones_like(mat_project))  # else, use ones
        

        iteration_new = iteration + 1
        
        # Flatten mat for the updated state.
        mat_new = tf.reshape(mat, [batch_size, -1])
        return norm_ratio, [mat_new, iteration_new]

# get avg jacobian
class CumulativeJacobian(AbstractRNNCell):
    """
    An RNN cell that computes a cumulative average of incoming Jacobian matrices.
    
    For each time step:
      - If iteration < self.start, the cell outputs the identity matrix (flattened).
      - Otherwise, it updates its cumulative average A with the new Jacobian J via:
          A_new = (A * count + J) / (count + 1)
        where count = iteration - self.start.
        
    The state is a list containing:
      1. The current cumulative average (flattened), of shape (batch, (2*nUnit)^2),
      2. The iteration counter, of shape (batch, 1).
    """
    def __init__(self, start=0, nUnit=0, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Returns initial state:
          - cumulative average: identity matrix of size (2*nUnit, 2*nUnit), flattened,
          - iteration counter: zeros.
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        # Create identity matrix for each batch element.
        I = tf.zeros((batch_size, 2 * self.nUnit, 2 * self.nUnit), dtype=dtype)
        I_flat = tf.reshape(I, [batch_size, -1])
        iteration = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [I_flat, iteration]

    @property
    def state_size(self):
        # State: [flattened cumulative average, iteration] 
        return [(2 * self.nUnit) ** 2, 1]

    @property
    def output_size(self):
        # The output is the flattened cumulative average.
        return (2 * self.nUnit) ** 2

    @tf.function
    def call(self, inputs, states):
        """
        Args:
          inputs: A flattened Jacobian tensor of shape (batch, 4*nUnit^2), where 4*nUnit^2 = (2*nUnit)^2.
          states: A list [cum_flat, iteration] representing the current cumulative average (flattened)
                  and the iteration counter.
        
        Returns:
          output: The updated (or non-updated) cumulative average (flattened), of shape (batch, (2*nUnit)^2).
          new_states: A list [new_cum_flat, new_iteration].
        """
        # Unpack state.
        cum_flat = states[0]
        iteration = states[1]  # shape: (batch, 1)
        batch_size = tf.shape(cum_flat)[0]
        
        # Reshape the cumulative average and the incoming Jacobian back into square matrices.
        cum = tf.reshape(cum_flat, [batch_size, 2 * self.nUnit, 2 * self.nUnit])
        J = tf.reshape(inputs, [batch_size, 2 * self.nUnit, 2 * self.nUnit])
        
        # Assume that all batch elements have the same iteration number.
        iter_scalar = tf.cast(iteration[0, 0], tf.int32)
        
        # Define branch to update cumulative average.
        def update_average():
            # Compute the count as the number of updates so far.
            count = tf.cast(iter_scalar - self.start, tf.float32)
            # Incremental averaging: new_avg = (old_avg * count + J) / (count + 1)
            new_avg = (tf.cast(cum, tf.float32) * count + tf.cast(J, tf.float32)) / (count + 1.0)
            return new_avg
        
        # If iter_scalar is less than self.start, no update is done.
        def no_update():
            return cum
        
        new_cum = tf.cond(tf.less(iter_scalar, self.start), no_update, update_average)
        iteration_new = iteration + 1
        
        new_cum_flat = tf.reshape(new_cum, [batch_size, -1])
        return new_cum_flat, [new_cum_flat, iteration_new]





class EigOrSingJacobian(AbstractRNNCell):
    """
    An RNN cell that computes eigenvalues or singular values of the input Jacobian on each time step.
    
    Behavior:
      - If iteration < self.start:
           The cell outputs a default vector (ones) as if the Jacobian were the identity matrix.
      - Otherwise:
           If calc_eig is True: computes the eigenvalues of the Jacobian,
           sorts them in descending order by their absolute values, and returns them.
           If calc_eig is False: computes the singular values of the Jacobian,
           sorts them in descending order (they are nonnegative) and returns them.
           
    The state is a single tensor containing an iteration counter of shape (batch, 1).
    
    The output has shape (batch, 2*nUnit), i.e. one value per eigenvalue or singular value.
    """
    def __init__(self, start=0, nUnit=0, calc_eig=True, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit
        self.calc_eig = calc_eig

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        # Only an iteration counter is needed.
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [initial_iteration]

    @property
    def state_size(self):
        return [1]

    @property
    def output_size(self):
        return 2 * self.nUnit

    @tf.function
    def compute_sorted_eig(self, J):
        # Compute eigenvalues; they are complex.
        eig_vals = tf.linalg.eigvals(J)
        abs_eig = tf.abs(eig_vals)
        sorted_indices = tf.argsort(abs_eig, direction='DESCENDING', stable=True)
        sorted_eig = tf.gather(eig_vals, sorted_indices, batch_dims=1)
        return sorted_eig

    @tf.function
    def compute_sorted_singular(self, J):
        s = tf.linalg.svd(J, compute_uv=False)
        s_sorted = tf.sort(s, direction='DESCENDING')
        return s_sorted

    @tf.function
    def call(self, inputs, states):
        """
        Args:
          inputs: A flattened Jacobian tensor of shape (batch, 4*nUnit^2).
          states: A list with one element, the iteration counter of shape (batch, 1).
          
        Returns:
          output: A tensor of shape (batch, 2*nUnit) containing either eigenvalues (if calc_eig is True)
                  or singular values (if calc_eig is False), sorted in descending order by absolute value.
          new_states: The updated state (iteration counter incremented by 1).
        """
        iteration = states[0]  # shape: (batch, 1)
        batch_size = tf.shape(iteration)[0]
        
        # Reshape the flattened Jacobian to (batch, 2*nUnit, 2*nUnit)
        J = tf.reshape(inputs, [batch_size, 2 * self.nUnit, 2 * self.nUnit])
        
        # Extract a scalar from the first batch element.
        iter_scalar = tf.cast(iteration[0, 0], tf.int32)
        
        # Determine the output dtype based on whether we're computing eigenvalues or singular values.
        out_dtype = tf.complex64 if self.calc_eig else tf.float32

        def true_branch():
            if self.calc_eig:
                result = self.compute_sorted_eig(J)
            else:
                result = self.compute_sorted_singular(J)
            # Ensure the result is cast to the desired dtype.
            return tf.cast(result, out_dtype)

        def false_branch():
            # Default output: ones, using the same shape and dtype.
            return tf.ones((batch_size, 2 * self.nUnit), dtype=out_dtype)

        output = tf.cond(tf.greater_equal(iter_scalar, self.start),
                         true_branch,
                         false_branch)
        new_iteration = iteration + 1
        return output, [new_iteration]



class Noise_jacob(AbstractRNNCell):
    """
    This cell propagates a set of fixed test vectors (tangent directions) and computes 
    the norm ratios after applying the Jacobian. We assume the test matrix (test_mat) 
    is provided in shape (2*nUnit, ncol), meaning that its columns are the initial tangent vectors.
    
    The cell’s state consists of:
      - mat: a tensor of shape (batch, 2*nUnit, ncol) containing the current tangent vectors (flattened in state)
      - iteration: a counter (batch, 1)
    """
    def __init__(self, start=0, nUnit=0, sing_or_eig=True, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit
        self.sing_or_eig=sing_or_eig



    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Initializes the state:
          - mat: Broadcasted test_mat with shape (batch, 2*nUnit, ncol) and normalized (each column is unit norm)
          - iteration: A counter set to zeros (batch, 1)
        """
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]

        # Initial iteration counter: zeros of shape (batch, 1)
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.float32)

        return [initial_iteration]
    
    @property
    def state_size(self):
        # The state consists of the flattened mat (shape: 2*nUnit * ncol) and the iteration (shape: 1)
        return [1]
    
    @property
    def output_size(self):
        # We output one value per column of the Jacobian (rank)
        return self.rank
    
    @tf.function
    def call(self, inputs, states):
        """
        On each time step:
          - Reshape the state to obtain mat of shape (batch, 2*nUnit, ncol)
          - Reshape the flattened Jacobian (inputs) to shape (batch, 2*nUnit, 2*nUnit)
          - Compute the product: mat_after = tf.matmul(J, mat), obtaining (batch, 2*nUnit, ncol)
          - Compute the norm of each updated column vector (norm over axis=1)
          - Optionally update the state if the iteration counter exceeds a threshold.
          - Return norm_ratio (shape: batch x ncol) and updated state.
        """
        # Unpack states: mat is stored in flattened form and iteration is a counter.
        iteration = states[0]
        batch_size = tf.shape(inputs)[0]

        
        # Reshape the flattened Jacobian to a square matrix (batch, 2*nUnit, 2=rank)
        Jacobian_flat = inputs
        J = tf.reshape(Jacobian_flat, [batch_size, 2 * self.nUnit, -1])
        self.rank=tf.shape(J)[2]
        
        if iteration[0,0]>self.start:
            singular_val=tf.linalg.svd(J, compute_uv=False)
        else:
            singular_val=tf.zeros((batch_size, self.rank,),dtype=tf.float32)# (batch,rank)


        iteration_new = iteration + 1
        
        return singular_val, [iteration_new]


class Get_norm_ratio_2(tf.keras.layers.AbstractRNNCell):
    def __init__(self, start=0, nUnit=0, test_mat=None, update_state=True, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.nUnit = nUnit
        self.update_state = update_state
        if test_mat is None:
            self.test_mat = tf.ones([self.nUnit * 2, 1], dtype=tf.float32)
            self.ncol = 1
        else:
            self.test_mat = test_mat
            # Convert the second dimension to an integer if possible.
            self.ncol = int(tf.shape(self.test_mat)[1])
            
    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        """
        Returns the initial state for the RNN.
        For stateful RNN layers, either `inputs` is provided, or an explicit batch_size must be given.
        """
        # If inputs are provided, infer batch size; otherwise require batch_size argument.
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
        elif batch_size is None:
            raise ValueError("For stateful RNN, batch_size must be provided either via inputs or explicitly.")
        
        # Expand test_mat along the batch dimension and tile it.
        # Original shape of test_mat: (2*nUnit, ncol)
        state = tf.expand_dims(self.test_mat, axis=0)  # shape: (1, 2*nUnit, ncol)
        state = tf.tile(state, [batch_size, 1, 1])       # shape: (batch_size, 2*nUnit, ncol)
        
        # Normalize each state matrix along axis=1.
        norm = tf.norm(state, axis=1, keepdims=True)     # shape: (batch_size, 1, ncol)
        norm = tf.where(norm > 0, norm, tf.ones_like(norm))
        state_norm = state / norm                        # shape: (batch_size, 2*nUnit, ncol)
        
        # Flatten the normalized state: each batch element becomes a vector.
        state_norm_flat = tf.reshape(state_norm, [batch_size, -1])
        state_norm_flat=tf.cast(state_norm_flat,dtype=tf.float32)
        
        # Initialize iteration counter: one per batch element.
        initial_iteration = tf.zeros((batch_size, 1), dtype=tf.float32)
        return [state_norm_flat, initial_iteration]
    
    @property
    def state_size(self):
        # The state is a list: [flattened matrix state, iteration counter].
        # Flattened matrix has 2*nUnit * ncol elements, and the iteration is a scalar per sample.
        return [2 * self.nUnit * self.ncol, 1]
    
    @property
    def output_size(self):
        # The output is the final norm per batch element, which has shape (ncol,).
        return self.ncol
    
    @tf.function
    def call(self, inputs, states):
        """
        inputs: expected to be a flattened Jacobian.
        states: list containing the flattened state matrix and the iteration count.
        """
        batch_size = tf.shape(inputs)[0]
        state_mat_flat, iteration = states
        
        # Reshape the flattened state back to a matrix form.
        state_mat = tf.reshape(state_mat_flat, [batch_size, 2 * self.nUnit, self.ncol])
        
        # Reshape the inputs to form the square Jacobian matrix.
        j_flat = inputs
        J = tf.reshape(j_flat, [batch_size, 2 * self.nUnit, 2 * self.nUnit])
        J=tf.ones_like(J,dtype=tf.float32)
        
        # Multiply the Jacobian by the state matrix.
        mat_after = tf.matmul(J, state_mat)  # shape: (batch_size, 2*nUnit, ncol)
        
        # Compute the norm of the resulting matrix along axis=1.
        norm_after = tf.norm(mat_after, axis=1, keepdims=True)  # shape: (batch_size, 1, ncol)
        # Normalize the result.
        mat_after_norm = mat_after / tf.where(norm_after > 0, norm_after, tf.ones_like(norm_after))
        
        # Determine whether the iteration counter exceeds the start threshold for each batch item.
        iteration_bool = tf.greater(iteration, self.start)  # shape: (batch_size, 1), bool
        
        # Prepare the final norm output:
        # If iteration > start, use the computed norm; otherwise, output ones.
        norm_after_reshaped = tf.reshape(norm_after, [batch_size, self.ncol])
        final_norm = tf.where(tf.squeeze(iteration_bool, axis=-1),
                              norm_after_reshaped,
                              tf.ones([batch_size, self.ncol], dtype=tf.float32))
        
        # Update the state matrix if update_state is True:
        # We first build a condition for each element by broadcasting iteration_bool to match state_mat.
        condition = tf.logical_and(iteration_bool, tf.constant(self.update_state))
        condition_full = tf.broadcast_to(condition, tf.shape(state_mat))
        mat_updated = tf.where(condition_full, mat_after_norm, state_mat)
        
        # Flatten the updated matrix for the next state.
        mat_updated_flat = tf.reshape(mat_updated, [batch_size, -1])
        
        # Increment the iteration count.
        iteration = iteration + 1
        
        return final_norm, [mat_updated_flat, iteration]

 

class LyapunovRNNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, n_unit, ncol, epsilon=1e-12, **kwargs):
        """
        Parameters:
          n_unit: Integer number of units such that the matrix dimension is 2*n_unit.
          ncol: Number of columns in the matrix.
          epsilon: Small constant to avoid division by zero.
        """
        super(LyapunovRNNCell, self).__init__(**kwargs)
        self.n_unit = n_unit
        self.ncol = ncol
        self.epsilon = epsilon

    @property
    def state_size(self):
        # The state is a flattened version of a (2*n_unit x ncol) matrix.
        return 2 * self.n_unit * self.ncol

    @property
    def output_size(self):
        # The output is a ratio for each of the ncol directions.
        return self.ncol

    def call(self, inputs, states):
        """
        Parameters:
          inputs: Tensor of shape (batch, (2*n_unit)*(2*n_unit)) representing a flattened Jacobian.
          states: List with one tensor representing the previous state (flattened matrix of shape (batch, 2*n_unit*ncol)).
        
        Returns:
          output: Tensor of shape (batch, ncol) containing the ratio of the norm after multiplication
                  to the norm before multiplication.
          new_states: List containing the new state tensor (flattened normalized matrix).
        """
        batch_size = tf.shape(inputs)[0]
        # Reshape input Jacobian to (batch, 2*n_unit, 2*n_unit)
        J = tf.reshape(inputs, (batch_size, 2 * self.n_unit, 2 * self.n_unit))
        
        # Recover the matrix from the flattened state; shape becomes (batch, 2*n_unit, ncol)
        U = tf.reshape(states[0], (batch_size, 2 * self.n_unit, self.ncol))
        
        # Multiply the Jacobian with the current matrix: (batch, 2*n_unit, ncol)
        new_U = tf.matmul(J, U)
        
        # Compute the Euclidean norm for each column before and after multiplication.
        # For a given sample, U[:, j] is a vector of length 2*n_unit.
        norm_old = tf.norm(U, axis=1)      # Shape: (batch, ncol)
        norm_new = tf.norm(new_U, axis=1)    # Shape: (batch, ncol)
        
        # Compute the ratio of the norms, with a small epsilon for numerical stability.
        ratio = norm_new / (norm_old + self.epsilon)
        
        # Normalize the new matrix by dividing each column by its norm.
        # Use tf.expand_dims to allow division along the vector (2*n_unit) dimension.
        new_U_normalized = new_U / (tf.expand_dims(norm_new, axis=1) + self.epsilon)
        
        # Flatten the normalized matrix to (batch, 2*n_unit * ncol) so it can serve as the state in the next iteration.
        new_state = tf.reshape(new_U_normalized, (batch_size, 2 * self.n_unit * self.ncol))
        
        return ratio, [new_state]


#%% get running correlation
class Running_correlation(AbstractRNNCell):
    def __init__(self, nUnit,**kwargs):
        super().__init__(**kwargs)
        self.nUnit = nUnit

    def get_initial_state(self, inputs=None, batch_size=None, dtype=tf.float32):
        # Only an iteration counter is needed.
        if batch_size is None:
            batch_size = tf.shape(inputs)[0]
        added_noise_A = tf.zeros((batch_size, self.nUnit), dtype=tf.float32)
        added_noise_B = tf.zeros((batch_size, self.nUnit), dtype=tf.float32)
        return [added_noise_A, added_noise_B]

    @property
    def state_size(self):
        return [self.nUnit,self.nUnit]
    

    @property
    def output_size(self):
        return 2

    @tf.function
    def pearson_corr(self,x, y, axis=1, eps=1e-8):
        # x, y: [batch_size, nUnit]
        # 1) compute means per sample
        x_mean = tf.reduce_mean(x, axis=axis, keepdims=True)
        y_mean = tf.reduce_mean(y, axis=axis, keepdims=True)
    
        # 2) demean
        xm = x - x_mean
        ym = y - y_mean
    
        # 3) numerator = sum((x - mean_x)*(y - mean_y))
        cov = tf.reduce_sum(xm * ym, axis=axis)        # shape [batch_size]
    
        # 4) denominator = sqrt(sum((x - mean_x)^2) * sum((y - mean_y)^2))
        x_var = tf.reduce_sum(tf.square(xm), axis=axis)
        y_var = tf.reduce_sum(tf.square(ym), axis=axis)
        denom = tf.sqrt(x_var * y_var) + eps
        return cov / denom   # shape [batch_size]

    @tf.function
    def call(self, inputs, states):
        added_noise_A, added_noise_B = states  # shape: (batch, nUnit)
        batch_size = tf.shape(added_noise_A)[0]
        
        act_A, act_B, noise_A_t, noise_B_t, binary_A, binary_B = tf.split(
            inputs,
            [self.nUnit, self.nUnit,    # po_A, po_B
             self.nUnit, self.nUnit,    # noise_add_A, noise_add_B
             1, 1],                     # bin_A, bin_B
            axis=-1
        )
        # Update the running noise where the binary flag is 1
        mask_A = tf.cast(binary_A > 0.5, tf.bool)   # shape (batch,1)
        mask_B = tf.cast(binary_B > 0.5, tf.bool)
        added_noise_A = tf.where(mask_A, noise_A_t, added_noise_A)  # (batch,n_unit)
        added_noise_B = tf.where(mask_B, noise_B_t, added_noise_B)
        
        # obtain correlation
        corr_A=self.pearson_corr(act_A, added_noise_A)# shape: (batch,)
        corr_B=self.pearson_corr(act_B, added_noise_B)# shape: (batch,)
        
        # Stack into shape (batch, 2)
        output = tf.stack([corr_A, corr_B], axis=1)        
        return output, [added_noise_A, added_noise_B]


#%% 
# get diagonal element
class DiagonalLayer(tf.keras.layers.Layer):
    def __init__(self, n, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def call(self, inputs):
        # inputs: (batch_size, n*n) or even (batch_size, timesteps, n*n)
        shape = tf.shape(inputs)
        # last dim is n*n
        # first dims are [batch_size] or [batch_size, timesteps]
        leading = shape[:-1]
        flat    = tf.reshape(inputs, tf.concat([[-1], [self.n, self.n]], axis=0))
        diag    = tf.linalg.diag_part(flat)        # (batch_size*timesteps, n)
        return tf.reshape(diag, tf.concat([leading, [self.n]], axis=0))
       

class LE_axis_diff(tf.keras.layers.Layer):
    def __init__(self, nUnit_all, u, v, **kwargs):
        """
        u:(nUnit,ncol)
        v:(nUnit,ncol)
        """
        super().__init__(**kwargs)
        self.nUnit_all = int(nUnit_all)
        self.nUnit=int(nUnit_all/2)
        u=tf.cast(u, tf.float32)
        v=tf.cast(v, tf.float32)
        uv_cat=tf.concat([u,v],0) # (2*nUnit,ncol)
        uv_norm=tf.linalg.norm(uv_cat, axis=0, keepdims=True)#(1,ncol)
        self.u=u/uv_norm
        self.v=v/uv_norm
        self.u_pv=tf.concat([self.u,self.v], 0) # (2*nUnit,ncol)
        self.u_mv=tf.concat([self.u,-self.v], 0) # (2*nUnit,ncol)
        self.u_pv/=tf.linalg.norm(self.u_pv, axis=0, keepdims=True)
        self.u_mv/=tf.linalg.norm(self.u_mv, axis=0, keepdims=True)
        self.ncol=int(tf.shape(u)[1])
        

    def call(self, inputs):
        # inputs: [..., nUnit_all * nUnit_all]
        # 1) figure out the “batch” dims
        input_shape = tf.shape(inputs)
        lead_dims   = input_shape[:-1]              # could be [batch] or [batch, timesteps], etc.
        flat_batch  = tf.reduce_prod(lead_dims)     # scalar: total # of slices

        # 2) reshape into [flat_batch, nUnit_all, nUnit_all]
        x_flat = tf.reshape(
            inputs,
            tf.concat([[flat_batch, self.nUnit_all, self.nUnit_all]], axis=0)
        )

        # 3) compute both projections
        # project via tensordot (no huge broadcast)
        # result: [flat_batch, nUnit_all, ncol]
        ax_flat = tf.tensordot(x_flat, self.u_pv, axes=[[2], [0]])
        df_flat = tf.tensordot(x_flat, self.u_mv, axes=[[2], [0]])

        # 4) take inner product along unit‐axis
        le_ax   = tf.reduce_sum(self.u_pv * ax_flat, axis=1)  # [flat_batch, ncol]
        le_diff = tf.reduce_sum(self.u_mv * df_flat, axis=1)  # [flat_batch, ncol]

        # 5) concat and restore original leading dims
        out_flat = tf.concat([le_ax, le_diff], axis=-1)       # [flat_batch, 2*ncol]
        out_shape = tf.concat([lead_dims, [2 * self.ncol]], axis=0)
        return tf.reshape(out_flat, out_shape) # [batch, timesteps, 2*ncol]



class Jxy_val(tf.keras.layers.Layer):
    def __init__(self, nUnit_all, u, v, **kwargs):
        """
        u:(nUnit,ncol)
        v:(nUnit,ncol)
        """
        super().__init__(**kwargs)
        self.nUnit_all = int(nUnit_all)
        self.nUnit=int(nUnit_all/2)
        u=tf.cast(u, tf.float32)
        v=tf.cast(v, tf.float32)
        uv_cat=tf.concat([u,v],0) # (2*nUnit,ncol)
        uv_norm=tf.linalg.norm(uv_cat, axis=0, keepdims=True)#(1,ncol)
        self.uv_norm=uv_norm
        self.u=u/uv_norm #divide by the norm of the concatenated vector
        self.v=v/uv_norm
        self.uo=tf.concat([self.u,tf.zeros_like(self.v)], 0) # (2*nUnit,ncol)
        self.ov=tf.concat([tf.zeros_like(self.u),self.v], 0) # (2*nUnit,ncol)
        # record the norm of the vector
        self.uo_norm=tf.linalg.norm(self.uo, axis=0, keepdims=True)#(1,ncol)
        self.ov_norm=tf.linalg.norm(self.ov, axis=0, keepdims=True)
        self.ncol=int(tf.shape(u)[1])
        

    def call(self, inputs):
        # inputs: [..., nUnit_all * nUnit_all]
        # 1) figure out the “batch” dims
        input_shape = tf.shape(inputs)
        lead_dims   = input_shape[:-1]              # could be [batch] or [batch, timesteps], etc.
        flat_batch  = tf.reduce_prod(lead_dims)     # scalar: total # of slices

        # 2) reshape into [flat_batch, nUnit_all, nUnit_all]
        x_flat = tf.reshape(
            inputs,
            tf.concat([[flat_batch, self.nUnit_all, self.nUnit_all]], axis=0)
        )

        # 3) compute both projections
        # project via tensordot (no huge broadcast)
        # result: [flat_batch, nUnit_all, ncol]
        uo_flat = tf.tensordot(x_flat, self.uo, axes=[[2], [0]])
        ov_flat = tf.tensordot(x_flat, self.ov, axes=[[2], [0]])

        # 4) take inner product along unit‐axis
        uJu = tf.reduce_sum(self.uo * uo_flat, axis=1)  # [flat_batch, ncol]
        vJu = tf.reduce_sum(self.ov * uo_flat, axis=1)  # [flat_batch, ncol]
        uJv = tf.reduce_sum(self.uo * ov_flat, axis=1)  # [flat_batch, ncol]
        vJv = tf.reduce_sum(self.ov * ov_flat, axis=1)  # [flat_batch, ncol]

        # 5) concat and restore original leading dims
        out_flat = tf.concat([uJu, vJu, uJv, vJv], axis=-1)       # [flat_batch, 2*ncol]
        out_shape = tf.concat([lead_dims, [4 * self.ncol]], axis=0)
        return tf.reshape(out_flat, out_shape) # [batch, timesteps, 2*ncol]



"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
class QRDcell(Layer):
# input at each timepoint should be of the size (batch_size, nUnit, nUnit)
    def __init__(self,start=0, **kwargs):
        super(QRDcell, self).__init__(**kwargs)
        self.start=start


        
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # shape of inputs: (batch_size, nUnit, nUnit)
        self.nUnit=tf.shape(inputs)[-1]
        iteration = tf.zeros((batch_size, 1), dtype=tf.int32)
        return [tf.linalg.eye(self.nUnit, self.nUnit, batch_shape=[batch_size]),iteration]


    @property
    def state_size(self):
        return [tf.TensorShape([self.nUnit, self.nUnit]), tf.TensorShape([None,1])]
    
    def call(self, inputs, states):
        # inputs is  J_{n} (batch, nUnit, nUnit)
        # states is Q_{n-1} (batch, nUnit,nUnit). Q_{0} is identity matrix
        # Q_{n}R_{n} = J_{n}Q_{n-1}
        # Q_{n} is the next state. (Q_new)
        # R_{n} is the output. (R_new)
          
        J = inputs
        Q = states[0]
        iteration=states[1]
        
        if iteration[0]>self.start:
            Q_new, R_new = tf.linalg.qr(tf.matmul(J,Q))
        else:
            Q_new=Q
            R_new=tf.zeros_like(Q)
        iteration+=1
        return R_new, [Q_new, iteration]
    """
