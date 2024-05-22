import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

# Constants
c1 = 0.03366340416928765

def f(x):
    c2 = 4 / 3000
    term1 = np.log10(x / (1 - x))
    term2 = np.log10(2**(c2 / (1 - x)) - 1)
    return term1 - term2

def h(d,f):
    log_term = 0 if d < 1 else 1.5 * np.log10(d)
    c3=(0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*(10**(-3))
    return 0.2 * (log_term + d * c3) - np.log10(c1)

def find_beta(y):
    # Define the equation f(x) - y = 0
    def equation(x):
        return f(x) - y
    
    try:
        beta = bisect(equation, 0.1, 0.9)
        return beta
    except ValueError:
        return None

d_values = np.linspace(1, 1300, 1000)  

beta_values = []
for d in d_values:
    y_value = h(d,f=40)
    beta_value = find_beta(y_value)
    if beta_value is not None:
        beta_values.append(beta_value)
    else:
        beta_values.append(np.nan)  
plt.figure(figsize=(10, 6))
plt.plot(d_values, beta_values, label='$\\beta = f^{-1}(h(d))$', color='blue')
plt.xlabel('Distance $d$')
plt.ylabel('$\\beta$')
plt.title('Plot of $\\beta = f^{-1}(h(d))$ as a function of distance $d$')
plt.legend()
plt.grid(True)
plt.show()
