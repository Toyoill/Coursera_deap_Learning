import numpy as np

foods = [
    [56.0, 0.0, 4.4, 68.0],
    [1.2, 104.0, 52.0, 8.0],
    [1.8, 135.0, 99.0, 0.9]
]
foods = np.array(foods)

cal = foods.sum(axis=0)
print(cal)

percentage = 100 * foods/cal.reshape(1, 4)
print(percentage)
