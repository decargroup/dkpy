import numpy as np

array = np.array(
    [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
    ]
)

diagonal = np.diagonal(array, axis1=1, axis2=2)

print(diagonal)
print(diagonal / np.max(diagonal, axis=0))
