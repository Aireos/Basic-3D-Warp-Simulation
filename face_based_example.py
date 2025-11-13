
# pip install pybullet
# pip install pybullet_data

import pybullet as p
import time
import pybullet_data

def run_simulation():
    # 1. Connect to the PyBullet physics server in GUI mode
    # This automatically opens a visualization window.
    physics_client = p.connect(p.GUI)
    
    # 2. Add the default search path for standard assets (like the ground plane)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # 3. Set up the environment physics (e.g., gravity)
    p.setGravity(0, 0, -10) # Earth's gravity in the Z direction

    # 4. Load the ground plane (static object)
    plane_id = p.loadURDF("plane.urdf")

    # 5. Create and add dynamic objects (boxes and spheres)

    # --- Create a box (dynamic object) ---
    # Define the visual and collision shapes for a box with half-extents (0.5, 0.5, 0.5)
    box_half_extents = [0.5, 0.5, 0.5]
    box_col_shape_id = p.createCollisionShape(p.GEOMETRY_BOX, halfExtents=box_half_extents)
    box_vis_shape_id = p.createVisualShape(p.GEOMETRY_BOX, halfExtents=box_half_extents, rgbaColor=[1, 0, 0, 1]) # Red color

    box_mass = 1.0
    box_start_pos = [1, 0, 5] # Start 5 meters high
    box_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    
    # Create the rigid body in the physics world
    box_id = p.createMultiBody(box_mass, box_col_shape_id, box_vis_shape_id, box_start_pos, box_start_orientation)


    # --- Create a sphere (dynamic object) ---
    # Define the visual and collision shapes for a sphere with a radius of 0.5
    sphere_radius = 0.5
    sphere_col_shape_id = p.createCollisionShape(p.GEOMETRY_SPHERE, radius=sphere_radius)
    sphere_vis_shape_id = p.createVisualShape(p.GEOMETRY_SPHERE, radius=sphere_radius, rgbaColor=[0, 0, 1, 1]) # Blue color

    sphere_mass = 1.0
    sphere_start_pos = [-1, 0, 7] # Start 7 meters high
    
    # Create the rigid body in the physics world (orientation is ignored for perfect spheres)
    sphere_id = p.createMultiBody(sphere_mass, sphere_col_shape_id, sphere_vis_shape_id, sphere_start_pos)


    print("Starting simulation loop. Press Ctrl+C to exit.")

    # 6. Run the simulation loop
    while p.isConnected():
        # Step the simulation forward by a fixed time increment
        p.stepSimulation()
        
        # Optional: Add a short sleep to slow down the visualization for human viewing
        time.sleep(1./240.) 

    # 7. Disconnect when the loop ends (e.g., the window is closed)
    p.disconnect()

if __name__ == "__main__":
    run_simulation()
