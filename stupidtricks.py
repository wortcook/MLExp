import numpy as np

base_complex = np.array([[1, -1j], [1j, -1]])

print(base_complex)
print('='*20)
next = np.matmul(base_complex, base_complex)/2
print(next)
print('='*20)
next = np.matmul(next, base_complex)/2
print(next)
print('='*20)
next = np.matmul(next, base_complex)/2
print(next)
print('='*20)
next = np.matmul(next, base_complex)/2
print(next)
print('='*20)