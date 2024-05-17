import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x) with base-10 logarithms
def f(x):
    c1 = 4 / 3000
    term1 = np.log10(x / (1 - x))
    term2 = np.log10(2**(c1 / (1 - x)) - 1)
    return term1 - term2

# Define the derivative of the function f(x)
def f_prime(x):
    c1 = 4 / 3000
    term1 = 1 / (np.log(10) * x * (1 - x))
    term2 = - (2**(c1 / (1 - x)) * np.log(2) * (-c1 / (1 - x)**2)) / (np.log(10) * (2**(c1 / (1 - x)) - 1))
    return term1 + term2

# Generate x values in the range 0.1 < x < 0.9
x_range = np.linspace(0.1, 0.9, 1000)

# Calculate f(x) values
y_f = f(x_range)

# Calculate f'(x) values
y_f_prime = f_prime(x_range)
print("f(0.2)=",f(0.2))
print("f(0.8)",f(0.8))
# Plot the function f(x)
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_f, label='$f(x) = \log_{10}\left(\\frac{x}{1-x}\\right) - \log_{10}\left(2^{\\frac{4}{3000(1-x)}} - 1\\right)$', color='blue')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Plot of the function $f(x)$ for $0.1 < x < 0.9$')
plt.legend()
plt.grid(True)
plt.show()

# Plot the derivative f'(x)
plt.figure(figsize=(10, 6))
plt.plot(x_range, y_f_prime, label="$f'(x)$", color='red')
plt.xlabel('x')
plt.ylabel("$f'(x)$")
plt.title("Plot of the derivative $f'(x)$ for $0.1 < x < 0.9$")
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.show()
