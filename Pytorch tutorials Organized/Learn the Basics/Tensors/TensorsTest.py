import torch

# Step 1: Creating Tensors with Explicit Data Types
tensor_a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)  # Create tensor with float32 type
print("Tensor A:")
print(tensor_a)

tensor_b = torch.rand(2, 3)  # Random values between 0 and 1 (2x3 tensor)
tensor_c = torch.zeros(3, 3)  # Tensor filled with zeros (3x3 tensor)
tensor_d = torch.ones(2, 2)  # Tensor filled with ones (2x2 tensor)

print("\nRandom Tensor B:")
print(tensor_b)
print("\nZero Tensor C:")
print(tensor_c)
print("\nOnes Tensor D:")
print(tensor_d)

# Step 2: Tensor Operations
# Element-wise addition
result_add = tensor_a + tensor_d  # Adds corresponding elements of tensor_a and tensor_d
print("\nElement-wise Addition (A + D):")
print(result_add)

# Matrix multiplication
result_matmul = torch.matmul(tensor_a, tensor_d)  # Matrix multiplication of A and D
print("\nMatrix Multiplication (A @ D):")
print(result_matmul)

# Transpose of a tensor
transposed_tensor = tensor_a.t()  # Transpose of tensor_a
print("\nTransposed Tensor A:")
print(transposed_tensor)

# Reshaping a tensor
reshaped_tensor = tensor_b.view(3, 2)  # Reshape tensor_b from (2, 3) to (3, 2)
print("\nReshaped Tensor B:")
print(reshaped_tensor)