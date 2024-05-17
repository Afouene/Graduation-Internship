import numpy as np
import matplotlib.pyplot as plt

# Define a sample function
def f(x):
    return x**3 - x - 2

# Bisection method demonstration
def bisection_demo(a, b, tol=1e-5):
    points = []
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        points.append((a, b, c, f(a), f(b), f(c)))
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return points

# Initial interval
a, b = 1, 2

# Perform bisection method and get points
points = bisection_demo(a, b)

# Plot the function
x = np.linspace(0, 3, 400)
y = f(x)
plt.plot(x, y, label='$f(x) = x^3 - x - 2$')

# Plot bisection steps
for i, (a, b, c, fa, fb, fc) in enumerate(points):
    plt.plot([a, a], [0, fa], 'r--')
    plt.plot([b, b], [0, fb], 'r--')
    plt.plot([c, c], [0, fc], 'go')
    plt.annotate(f'Step {i+1}', (c, fc), textcoords="offset points", xytext=(0,10), ha='center')

plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Bisection Method Visualization')
plt.legend()
plt.grid(True)
plt.show()
