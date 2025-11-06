import numpy as np
import warp as wp
import pyglet
# Import only necessary core GL functions and constants
from pyglet.gl import (
    glClearColor, glEnable, glViewport, glPointSize, glColor3f, glBegin, glEnd, 
    glVertex3f, GL_DEPTH_TEST, GL_POINTS, GL_LINES, GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT
)
from pyglet.math import Mat4, Vec3

# (Keep all your existing warp setup, kernel definitions, and array initializations)
# ... (wp.init(), device = wp.get_device("cuda:0"), @wp.kernel integrate(...), etc.) ...
# ... (wp_x, wp_v, wp_f, wp_w, sim_gravity, sim_dt are all defined) ...


wp.init()
device = wp.get_device("cuda:0") 
print(f"Using device: {device.name}")

@wp.kernel
def integrate(x: wp.array(dtype=wp.vec3),
              v: wp.array(dtype=wp.vec3),
              f: wp.array(dtype=wp.vec3),
              w: wp.array(dtype=float),
              gravity: wp.vec3,
              dt: float):  

   tid = wp.tid()  
   v1 = v[tid] + (f[tid] * w[tid] + gravity) * dt
   x[tid] = x[tid] + v1 * dt
   v[tid] = v1

num_particles = 4
sim_gravity = wp.vec3(0.0, -9.81, 0.0)
sim_dt = 0.01

np_x = np.zeros((num_particles, 3), dtype=np.float32)
np_v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
np_f = np.zeros((num_particles, 3), dtype=np.float32)
np_w = np.ones(num_particles, dtype=np.float32) 

wp_x = wp.from_numpy(np_x, dtype=wp.vec3, device=device)
wp_v = wp.from_numpy(np_v, dtype=wp.vec3, device=device)
wp_f = wp.from_numpy(np_f, dtype=wp.vec3, device=device)
wp_w = wp.from_numpy(np_w, dtype=float, device=device) 


# --- Visualization Setup (Modern Pyglet 2.0+) ---
window = pyglet.window.Window(width=800, height=600, caption='Warp Physics Simulation', resizable=True)
glClearColor(0.9, 0.9, 0.9, 1.0)
glEnable(GL_DEPTH_TEST)

# Use Pyglet 2.0 window properties for projection and view
# We define a custom view matrix (camera position)
view_matrix = Mat4.look_at(
    eye=Vec3(0, 5, 15),    # Camera position
    target=Vec3(0, 0, 0),  # Look at center
    up=Vec3(0, 1, 0)       # Up direction
)

@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    # Pyglet 2.0 handles the projection matrix internally if you set the 'projection' attribute
    window.projection = Mat4.perspective_projection(window.width, window.height, near=0.1, far=100.0, fov=65.0)
    return pyglet.event.EVENT_HANDLED

def update(dt):
    # This function is called every frame by pyglet
    
    # 1. Integrate Physics
    wp.launch(
        kernel=integrate,
        dim=num_particles,
        inputs=[wp_x, wp_v, wp_f, wp_w, sim_gravity, sim_dt]
    )
    
    wp.synchronize() 

# Schedule the update function
pyglet.clock.schedule_interval(update, sim_dt) 

@window.event
def on_draw():
    # Set the view matrix for every draw call (simulating camera movement)
    window.view = view_matrix 
    window.clear()
    
    # Get the particle positions as a NumPy array view
    positions_np = wp_x.numpy() 

    # Draw the particles as points
    glPointSize(5.0)
    glColor3f(1.0, 0.0, 0.0) 
    pyglet.graphics.draw(num_particles, GL_POINTS,
        ('v3f', positions_np.flatten()) 
    )
    
    # Draw a simple floor grid (simple lines)
    glColor3f(0.5, 0.5, 0.5, 1.0)
    glBegin(GL_LINES)
    for i in range(-10, 11):
        glVertex3f(float(i), 0.0, -10.0)
        glVertex3f(float(i), 0.0, 10.0)
        glVertex3f(-10.0, 0.0, float(i))
        glVertex3f(10.0, 0.0, float(i))
    glEnd()


# Run the application loop
print("\nStarting visualization window using Pyglet 2.0. Close the window to exit.")
pyglet.app.run()
