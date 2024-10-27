#Homework 1

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaussian(x, A1, mu1, sigma1) + gaussian(x, A2, mu2, sigma2)


data = pd.read_csv('C:\\Users\\user\\Desktop\\CPS\\hist2.csv')  
x = data.iloc[:, 0] 
y = data.iloc[:, 1]  


plt.bar(x, y, color='gray', alpha=0.7, label='Data Histogram', width=0.0815)


initial_guess_A = [max(y), x[np.argmax(y)], 0.5]  
initial_guess_B = [max(y)/2, x[np.argmax(y)] + 1, 0.5]  
initial_guess = initial_guess_A + initial_guess_B


popt, _ = curve_fit(double_gaussian, x, y, p0=initial_guess)


x_fit = np.linspace(min(x), max(x), 1000)
y_fit_A = gaussian(x_fit, popt[0], popt[1], popt[2])  
y_fit_B = gaussian(x_fit, popt[3], popt[4], popt[5])  
y_fit_total = double_gaussian(x_fit, *popt)  


area_A = np.trapz(y_fit_A, x_fit)
area_B = np.trapz(y_fit_B, x_fit)
ratio = area_A / area_B


print(f'Area of A: {area_A}')
print(f'Area of B: {area_B}')
print(f'Ratio of A to B: {ratio}')


plt.plot(x_fit, y_fit_A, 'r--', label='Gaussian Fit A')
plt.plot(x_fit, y_fit_B, 'y--', label='Gaussian Fit B')
plt.plot(x_fit, y_fit_total, 'b-', label='Total Fit')
plt.xlabel('Energy')
plt.ylabel('Count')
plt.legend()
plt.title('Gaussian Fit of Two Particle Spectra with Histogram')
plt.grid(True)
plt.show()









