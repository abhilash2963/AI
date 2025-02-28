import numpy as np
import matplotlib.pyplot as plt



X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 1.9, 4.1, 5.2])


n = len(X)


sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xx = np.sum(X**2)
sum_xy = np.sum(X * y)


m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
b = (sum_y - m * sum_x) / n


print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")


y_pred = m * X + b


mse = np.mean((y - y_pred)**2)


ss_total = np.sum((y - np.mean(y))**2)
ss_residual = np.sum((y - y_pred)**2)
r2 = 1 - (ss_residual / ss_total)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Simple Linear Regression')
plt.show()
