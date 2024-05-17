import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# Constants
c1 = 0.03366340416928765
c3 = 0.004133836796896648

# Define the function f(x) with base-10 logarithms
def f(x):
    c1_inner = 4 / 3000
    term1 = np.log10(x / (1 - x))
    term2 = np.log10(2**(c1_inner / (1 - x)) - 1)
    return term1 - term2

# Define the function h(d) with the given condition
def h(d):
    log_term = 0 if d < 1 else 1.5 * np.log10(d)
    return 0.2 * (log_term + d * c3) - np.log10(c1)

# Define the inverse function using bisection method
def find_beta(y):
    # Define the equation f(x) - y = 0
    def equation(x):
        return f(x) - y
    
    try:
        # Use bisection method to find the root in the interval 0.1 < x < 0.9
        beta = bisect(equation, 0.1, 0.9)
        return beta
    except ValueError:
        # If bisect fails, return None to indicate out of range
        return None

# Generate distance values
d_values = np.linspace(0, 1300, 1000)  # Use a range of d from 1 to 1300

# Calculate beta values
beta_values = []
for d in d_values:
    y_value = h(d)
    beta_value = find_beta(y_value)
    if beta_value is not None:
        beta_values.append(beta_value)
    else:
        beta_values.append(np.nan)  # Use NaN for out of range values

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(d_values, beta_values, label='$\\beta = f^{-1}(h(d))$', color='blue')
plt.xlabel('Distance $d$')
plt.ylabel('$\\beta$')
plt.title('Plot of $\\beta = f^{-1}(h(d))$ as a function of distance $d$')
plt.legend()
plt.grid(True)
plt.show()
