import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
y_tan = np.tan(x)

# Plot the points using matplotlib
plt.plot(x, y_sin, color='red')
plt.plot(x, y_cos, 'blue')
plt.plot(x, y_tan, 'green')
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine', "tan hello"])
plt.show()