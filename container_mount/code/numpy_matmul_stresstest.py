import numpy as np
import time
import tensorflow as tf

def tf_matmul_func(M):
    N = tf.matmul(M, M)
    return N

n = 35000
#M_np = np.random.normal(size=(n, n))
M_np = np.random.normal(size=(n, n)).astype(dtype=np.float32)
M = tf.convert_to_tensor(M_np)
print(M[n-1,n-1])

start_time_s = time.time()
N = tf_matmul_func(M)
print(N[n-1,n-1])
end_time_s = time.time()

diff_time_s = end_time_s - start_time_s
print('Matmul time in sec: ' + str(round(diff_time_s, 1)))
