import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# Generate initial points
num_points = 10000

def generate_directional(angle,length,sharpness)->[int,int]:
    d=np.random.random()*np.random.random()*length*length
    l=np.random.normal(0)*sharpness
    dx=np.cos(angle)
    dy=np.sin(angle)
    x =dx*d + l*dy
    y=dy*d -l*dx
    return x, y

def generate_2D_random_points(angle,length,sharpness)->[int,int]:
    d1=generate_directional(angle,length,sharpness)
    d2=generate_directional(0,1,0.1)

    x =d1[0]+d2[0]
    y=d1[1]+d2[1]
    return x, y

def get_len2(s):
    l2=s[0]*s[0]+s[1]*s[1]
    return l2

def quantify(s, q):
    l=np.sqrt(s[0]*s[0]+s[1]*s[1])
    scale=np.round(l/q)*q
    return s[0]*scale,s[1]*scale

def update(event=None):
    w=1
    samples=[generate_2D_random_points(s1.val, s2.val, s3.val) for i in range(0,num_points) ]
    samples = [quantify(s,0.1) for s in samples]
    x = [s[0] for s in samples ]
    y = [s[1] for s in samples ]
    ax.clear()
    ax.hist2d(x, y,range=[[-w, w],[-w, w]] ,bins=200, cmap='plasma')
    ax.axhline(0, color='grey', linewidth=.5)
    ax.axvline(0, color='grey', linewidth=.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Random Points Probability Distribution')
    plt.draw()

# Create a plot
fig, ax = plt.subplots(figsize=(5, 5))
plt.subplots_adjust(bottom=0.25)

# Slider
s1_ax = plt.axes([0.15, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
s1 = Slider(s1_ax, 'Angle', -np.pi, np.pi, valinit=0)
s1.on_changed(update)

# Slider
s2_ax = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
s2 = Slider(s2_ax, 'Len', 0, 4, valinit=1)
s2.on_changed(update)

# Slider
s3_ax = plt.axes([0.15, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')
s3 = Slider(s3_ax, 'Sharp', 0, 1, valinit=0)
s3.on_changed(update)

# Initialize plot by calling update
update()

plt.show()
