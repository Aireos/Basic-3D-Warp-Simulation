import numpy as np # Import the NumPy library for efficient array manipulation in CPU memory
import warp as wp # Import the Warp library for high-performance physics simulations and GPU computing
import pyglet # Import the Pyglet library for creating a cross-platform window and handling events/rendering
# Import all legacy GL functions and required matrix utility functions from the Pyglet OpenGL bindings
from pyglet.gl import * 
try:
    # Attempt to import gluPerspective, necessary for setting up 3D camera perspective
    from pyglet.gl import gluPerspective
except ImportError:
    # Fallback import path for specific Pyglet/OS configurations
    from pyglet.gl.glu import gluPerspective


# Initialize the Warp library context and JIT compilation system
wp.init()
# Get a handle to the specific CUDA device (GPU) named "cuda:0"
device = wp.get_device("cuda:0") 
print(f"Using device: {device.name}")

# Define a Warp kernel function that will execute in parallel on the GPU threads
@wp.kernel
def integrate(x: wp.array(dtype=wp.vec3), # type: ignore # Input/Output: Array of particle positions (wp.vec3 floats)
              v: wp.array(dtype=wp.vec3), # type: ignore # Input/Output: Array of particle velocities (wp.vec3 floats)
              f: wp.array(dtype=wp.vec3), # type: ignore # Input: Array of forces applied to particles (wp.vec3 floats)
              w: wp.array(dtype=float),   # type: ignore # Input: Array of inverse mass values (float)
              gravity: wp.vec3, # type: ignore # Input: A constant 3D gravity vector (wp.vec3)
              dt: float):  # Input: The constant simulation timestep (float delta time)
   
   tid = wp.tid()  # Get the unique thread ID (index of the current particle being processed)
   # Update velocity using explicit Euler integration: v_new = v_old + (force * inv_mass + gravity) * dt
   v1 = v[tid] + (f[tid] * w[tid] + gravity) * dt
   # Update position using explicit Euler integration: x_new = x_old + v_new * dt
   x[tid] = x[tid] + v1 * dt
   # Store the newly calculated velocity back into the global velocity array
   v[tid] = v1

# Defines the total count of simulation particles (currently 1)
num_particles = 1

# Defines the constant 3D vector representing gravitational acceleration (m/s^2)
sim_gravity = wp.vec3(0.0, -9.81, 0.0)

# Defines the fixed simulation timestep duration in seconds (0.01s per physics update)
sim_dt = 0.01

# Creates a NumPy array (CPU memory) of shape (num_particles, 3) initialized to zeros, 
# ready to hold float32 position data.
np_x = np.zeros((num_particles, 3), dtype=np.float32)

# Creates a NumPy array (CPU memory) with specific initial float32 values for velocity data.
np_v = np.array([[1.0, 10.0, 1.0]], dtype=np.float32)

# Creates a NumPy array (CPU memory) for forces, initialized to zeros (forces are dynamic inputs).
np_f = np.zeros((num_particles, 3), dtype=np.float32)

# Creates a NumPy array (CPU memory) of ones for the inverse mass values (mass=1.0 kg for all particles).
np_w = np.ones(num_particles, dtype=np.float32) 

# Converts the NumPy position array from CPU memory to a Warp array in GPU memory, formatted as wp.vec3.
wp_x = wp.from_numpy(np_x, dtype=wp.vec3, device=device)

# Converts the NumPy velocity array from CPU memory to a Warp array in GPU memory, formatted as wp.vec3.
wp_v = wp.from_numpy(np_v, dtype=wp.vec3, device=device)

# Converts the NumPy force array from CPU memory to a Warp array in GPU memory, formatted as wp.vec3.
wp_f = wp.from_numpy(np_f, dtype=wp.vec3, device=device)

# Converts the NumPy inverse mass array from CPU memory to a Warp array in GPU memory, formatted as single floats.
wp_w = wp.from_numpy(np_w, dtype=float, device=device) 


# --- Camera Control Variables ---
# Camera angles for rotation around the target (orbit camera)
camera_yaw = -90.0  # Rotation angle in degrees around the Y (vertical) axis
camera_pitch = -20.0 # Rotation angle in degrees around the X (horizontal) axis
camera_distance = 250.0 # Distance of the camera from the origin (zoom level)
mouse_sensitivity = 0.2 # Multiplier for mouse movement speed during rotation
zoom_sensitivity = 2 # Multiplier for scroll wheel speed during zooming

# --- Visualization Setup (Legacy Pyglet 1.5.x) ---
# Creates the Pyglet window with specified dimensions, title, and resize capability
window = pyglet.window.Window(width=800, height=600, caption='Warp Physics Simulation', resizable=True)

# Sets the background color of the window when glClear is called (R, G, B, Alpha: light gray, 0.0 to 1.0 range)
glClearColor(0.9, 0.9, 0.9, 1.0)

# Enables depth testing within OpenGL, ensuring closer objects correctly obscure farther objects
glEnable(GL_DEPTH_TEST)

# This decorator registers the function as the event handler for when the window is resized
@window.event
def on_resize(width, height):
    # Sets the OpenGL viewport to cover the entire window area
    glViewport(0, 0, width, height)
    # Switches the current matrix mode to operate on the projection matrix (configuring the camera lens)
    glMatrixMode(GL_PROJECTION)
    # Resets the projection matrix to the identity matrix (clears previous camera lens settings)
    glLoadIdentity()
    # Sets up a perspective view: Field of View 65 deg, dynamically calculated Aspect Ratio, Near Clip Plane 0.1, Far Clip Plane 1000.0
    gluPerspective(65, width / float(height), 0.1, 1000.0) 
    # Switches the current matrix mode back to operating on the modelview matrix (configuring object/camera position)
    glMatrixMode(GL_MODELVIEW)
    # Returns True to signal that this custom handler has processed the event, preventing any default Pyglet handlers
    return pyglet.event.EVENT_HANDLED

# This function handles mouse drag events (used for rotation)
@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    """
    Handle mouse drag events for camera rotation (left button).
    """
    global camera_yaw, camera_pitch
    
    if buttons & pyglet.window.mouse.LEFT:
        # Update camera yaw angle based on horizontal mouse movement delta (dx) and sensitivity
        camera_yaw += dx * mouse_sensitivity
        # Update camera pitch angle based on vertical mouse movement delta (dy) and sensitivity (inverted Y axis)
        camera_pitch -= dy * mouse_sensitivity
        
        # Clamp pitch angle between -89 and 89 degrees to prevent the camera from flipping over
        if camera_pitch > 89.0:
            camera_pitch = 89.0
        if camera_pitch < -89.0:
            camera_pitch = -89.0

# This function handles mouse scroll events (used for zooming)
@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    """
    Handle mouse scroll events for zooming in and out by adjusting camera distance.
    """
    global camera_distance
    # Adjust distance based on vertical scroll amount (scroll_y, typically +/- 1.0) and sensitivity
    camera_distance -= scroll_y * zoom_sensitivity
    # Clamp distance to keep the camera within a reasonable range (1.0 to 100.0 units)
    if camera_distance < 1.0:
        camera_distance = 1.0
    if camera_distance > 300.0:
        camera_distance = 300.0

# --- Function to draw a 3D cube at the origin (size 1x1x1) ---
def draw_cube():
    # Define vertices for a unit cube centered at the origin
    vertices = [
        [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [-0.5,  0.5, -0.5], # Back face
        [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5], # Front face
        [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [-0.5, -0.5,  0.5], # Left face
        [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5], # Right face
        [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5, -0.5, -0.5], # Bottom face
        [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5]  # Top face
    ]
    
    # Use GL_QUADS for legacy OpenGL drawing
    glBegin(GL_QUADS)
    
    # Back face (Red)
    glColor3f(1.0, 0.0, 0.0)
    for v in vertices[0:4]: glVertex3f(*v)

    # Front face (Green)
    glColor3f(0.0, 1.0, 0.0)
    for v in vertices[4:8]: glVertex3f(*v)
    
    # Left face (Blue)
    glColor3f(0.0, 0.0, 1.0)
    for v in vertices[8:12]: glVertex3f(*v)

    # Right face (Yellow)
    glColor3f(1.0, 1.0, 0.0)
    for v in vertices[12:16]: glVertex3f(*v)

    # Bottom face (Magenta)
    glColor3f(1.0, 0.0, 1.0)
    for v in vertices[16:20]: glVertex3f(*v)

    # Top face (Cyan)
    glColor3f(0.0, 1.0, 1.0)
    for v in vertices[20:24]: glVertex3f(*v)
    
    glEnd()

# This function defines the logic that runs every simulation step
def update(dt):
    # Launch the 'integrate' kernel on the GPU for all particles
    wp.launch(
        kernel=integrate,
        dim=num_particles,
        inputs=[wp_x, wp_v, wp_f, wp_w, sim_gravity, sim_dt]
    )
    # Wait for the GPU to finish computation before the next frame is rendered
    wp.synchronize() 

# Schedules the 'update' function to be called every `sim_dt` seconds by the pyglet clock system
pyglet.clock.schedule_interval(update, sim_dt) 

# This decorator registers the function as the main drawing routine called every frame
@window.event
def on_draw():
    # Clears the color buffer (screen color) and the depth buffer (depth information)
    window.clear()
    
    # --- Apply Camera Transformation using mouse variables ---
    # Resets the current modelview matrix to the identity matrix
    glLoadIdentity()
    # Apply Zoom: Translates the entire scene away from the camera along the Z axis
    glTranslatef(0.0, 0.0, -camera_distance)
    # Apply Pitch: Rotates the scene around the X axis based on camera_pitch angle
    glRotatef(camera_pitch, 1.0, 0.0, 0.0)
    # Apply Yaw: Rotates the scene around the Y axis based on camera_yaw angle
    glRotatef(camera_yaw, 0.0, 1.0, 0.0)
    # The world/objects drawn after this point will be rendered with this camera perspective
    # -----------------------------------------------------

    glPushMatrix() # Save the current camera matrix
    glTranslatef(5.0, 0.5, 0.0) # Move the cube to a specific location (e.g., x=5)
    draw_cube()
    glPopMatrix()

    # Get the current particle positions from the GPU as a NumPy array view (efficient access)
    positions_np = wp_x.numpy() 

    # Draw the particles as points using legacy GL immediate mode
    glPointSize(5.0) # Set the size of the points in pixels
    glColor3f(1.0, 0.0, 0.0) # Set the drawing color to solid red (RGB 0.0-1.0)
    glBegin(GL_POINTS) # Signal the start of drawing points
    for i in range(num_particles):
        # Define a vertex for each particle's 3D position
        glVertex3f(positions_np[i, 0], positions_np[i, 1], positions_np[i, 2])
    glEnd() # Signal the end of drawing points
    
    # # Draw a simple floor grid using legacy GL lines
    # glColor3f(0.5, 0.5, 0.5) # Set the drawing color to solid gray
    # glBegin(GL_LINES) # Signal the start of drawing lines
    # for i in range(-10, 11):
    #     # Draw lines along the Z-axis (parallel to X-axis)
    #     glVertex3f(float(i), 0.0, -10.0)
    #     glVertex3f(float(i), 0.0, 10.0)
    #     # Draw lines along the X-axis (parallel to Z-axis)
    #     glVertex3f(-10.0, 0.0, float(i))
    #     glVertex3f(10.0, 0.0, float(i))
    # glEnd() # Signal the end of drawing lines

    # Draw a solid floor plane instead of a grid
    #          R     G    B
    glColor3f(0.0, 10.0, 0.0) # Light gray color = (0.8, 0.8, 0.8)
    glBegin(GL_QUADS)
    glVertex3f(-100.0, 0.0, -100.0) # Bottom left
    glVertex3f(100.0, 0.0, -100.0)  # Bottom right
    glVertex3f(100.0, 0.0, 100.0)   # Top right
    glVertex3f(-100.0, 0.0, 100.0)  # Top left
    glEnd()

# Print instructions to the console
print("\nStarting visualization window using Pyglet 1.5.x.")
print("Controls: Drag with Left Mouse Button to Rotate, Scroll Wheel to Zoom.")
# Run the application loop; this function blocks until the window is closed
pyglet.app.run()
