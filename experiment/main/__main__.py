import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

def generate_2D_random_points(mean, cov_matrix, num_points):
    x, y = np.random.multivariate_normal(mean, cov_matrix, num_points).T
    return x, y

def update(event=None):
    cov_value = slider.val
    cov_matrix = [[1, cov_value], [cov_value, 1]]
    x, y = generate_2D_random_points(mean, cov_matrix, num_points)
    ax.clear()
    ax.hist2d(x, y, bins=50, cmap='plasma')
    ax.axhline(0, color='white', linewidth=1.2)
    ax.axvline(0, color='white', linewidth=1.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Random Points Probability Distribution')
    plt.draw()

# Mean and covariance matrix for 2D Gaussian distribution
mean = (0, 0)
initial_cov_value = 0.5
cov_matrix = [[1, initial_cov_value], [initial_cov_value, 1]]

# Generate initial points
num_points = 10000

# Create a plot
fig, ax = plt.subplots(figsize=(10, 10))

# Slider
slider_ax = plt.axes([0.2, 0.025, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(slider_ax, 'Covariance', -1.0, 1.0, valinit=initial_cov_value)
slider.on_changed(update)

# Button
button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(button_ax, 'Update', color='lightgoldenrodyellow', hovercolor='0.975')
button.on_clicked(update)

# Initialize plot by calling update
update()

plt.show()
