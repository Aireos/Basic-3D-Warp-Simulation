import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import numpy as np
import warp as wp
from OpenGL.GL import *
from OpenGL.GLU import *
import math

# ==========================================================
# --- Simulation Parameters ---
# ==========================================================
num_particles = 128
screen_width, screen_height, screen_depth = 800, 600, 600
cube_size = 24.0

# ==========================================================
# --- Warp Kernel: Simple physics simulation (FULLY FIXED) ---
# ==========================================================
@wp.kernel
def update_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
    cube_size: float,
    bound_x: float,
    bound_y: float,
    bound_z: float
):
    tid = wp.tid()
    pos = positions[tid]
    vel = velocities[tid]

    # Apply gravity
    vel += wp.vec3(0.0, -9.8 * dt, 0.0)
    pos += vel * dt

    # Handle collisions between cubes (simple approximation)
    half = 0.5 * cube_size
    for j in range(positions.shape[0]):
        if j != tid:
            other_pos = positions[j]
            
            # FIX: Calculate dx, dy, dz as single float magnitudes
            dx = wp.abs(pos[0] - other_pos[0])
            dy = wp.abs(pos[1] - other_pos[1])
            dz = wp.abs(pos[2] - other_pos[2])
            
            overlap = (dx < cube_size) and (dy < cube_size) and (dz < cube_size)
            if overlap:
                # Minimal axis resolution
                if dx < dy and dx < dz:
                    sign = 1.0 if pos[0] > other_pos[0] else -1.0
                    vel = wp.vec3(sign * wp.abs(vel[0]), vel[1], vel[2]) # FIX: Access velocity components
                    pos += wp.vec3(sign * (cube_size - dx) * 0.5, 0.0, 0.0)
                elif dy < dz:
                    sign = 1.0 if pos[1] > other_pos[1] else -1.0
                    vel = wp.vec3(vel[0], sign * wp.abs(vel[1]), vel[2]) # FIX: Access velocity components
                    pos += wp.vec3(0.0, sign * (cube_size - dy) * 0.5, 0.0)
                else:
                    sign = 1.0 if pos[2] > other_pos[2] else -1.0
                    vel = wp.vec3(vel[0], vel[1], sign * wp.abs(vel[2])) # FIX: Access velocity components
                    pos += wp.vec3(0.0, 0.0, sign * (cube_size - dz) * 0.5)

    # Bounce off the walls
    # FIX: Access components for comparison and clamping
    if pos[0] < half or pos[0] > bound_x - half:
        vel = wp.vec3(-vel[0], vel[1], vel[2])
        pos = wp.vec3(wp.clamp(pos[0], half, bound_x - half), pos[1], pos[2])
    if pos[1] < half or pos[1] > bound_y - half:
        vel = wp.vec3(vel[0], -vel[1], vel[2])
        pos = wp.vec3(pos[0], wp.clamp(pos[1], half, bound_y - half), pos[2])
    if pos[2] < half or pos[2] > bound_z - half:
        vel = wp.vec3(vel[0], vel[1], -vel[2])
        pos = wp.vec3(pos[0], pos[1], wp.clamp(pos[2], half, bound_z - half))

    positions[tid] = pos
    velocities[tid] = vel
# ==========================================================
# --- Initialize Warp + particle data ---
# ==========================================================
wp.init()
init_positions = np.random.rand(num_particles, 3).astype(np.float32) * np.array([screen_width, screen_height, screen_depth], dtype=np.float32)
positions_gpu = wp.from_numpy(init_positions, dtype=wp.vec3, device="cuda")
init_vel = (np.random.rand(num_particles, 3) - 0.5) * 120.0
velocities_gpu = wp.from_numpy(init_vel.astype(np.float32), dtype=wp.vec3, device="cuda")
positions_cpu = wp.zeros(num_particles, dtype=wp.vec3, device="cpu")

# ==========================================================
# --- Pygame + OpenGL Setup ---
# ==========================================================
pygame.init()
pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)
glEnable(GL_DEPTH_TEST)
glDisable(GL_LIGHTING)
glClearColor(0.1, 0.1, 0.1, 1.0)
gluPerspective(45.0, screen_width / screen_height, 0.1, 5000.0)
glViewport(0, 0, screen_width, screen_height)

# ==========================================================
# --- Camera Settings ---
# ==========================================================
# FIX: Placed camera in front of the scene (negative Z)
cam_pos = np.array([screen_width / 2.0, screen_height / 2.0, -1000.0], dtype=float)
cam_pitch, cam_yaw = 0.0, 0.0
pygame.mouse.set_visible(False)
pygame.event.set_grab(True)
mouse_sensitivity, movement_speed = 0.1, 500.0
clock, running = pygame.time.Clock(), True

# ==========================================================
# --- Draw cube function (FIXED) ---
# ==========================================================
def draw_cube(size):
    half = size * 0.5
    vertices = [[ half,  half,  half], [ half, -half,  half], [-half, -half,  half], [-half,  half,  half],
                [ half,  half, -half], [ half, -half, -half], [-half, -half, -half], [-half,  half, -half]]
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    faces = [(0,1,2,3), (7,6,5,4), (0,4,5,1), (3,2,6,7), (0,3,7,4), (1,5,6,2)]
    glBegin(GL_QUADS)
    glColor3f(0.2, 0.8, 0.4)
    for face in faces:
        for vi in face:
            glVertex3f(*vertices[vi])
    glEnd()
    glColor3f(0.0, 0.0, 0.0)
    glBegin(GL_LINES)
    for edge in edges:
        for vi in edge:
            glVertex3f(*vertices[vi])
    glEnd()

# ==========================================================
# --- Optional: Draw bounding wireframe box (FIXED) ---
# ==========================================================
def draw_bounds(width, height, depth):
    vertices = [[0.0, 0.0, 0.0], [width, 0, 0], [width, height, 0], [0, height, 0],
                [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]]
    edges = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    glColor3f(0.3, 0.3, 0.9)
    glBegin(GL_LINES)
    for edge in edges:
        for vi in edge:
            glVertex3f(*vertices[vi])
    glEnd()

# ==========================================================
# --- Apply camera transformation (FIXED) ---
# ==========================================================
def apply_camera_transform():
    glRotatef(-cam_pitch, 1, 0, 0)
    glRotatef(-cam_yaw, 0, 1, 0)
    glTranslatef(*-cam_pos) # Unpack array

# ==========================================================
# --- Main Loop ---
# ==========================================================
while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()
    forward = np.array([-math.sin(math.radians(cam_yaw)), 0, -math.cos(math.radians(cam_yaw))], dtype=float)
    up_vector = np.array([0.0, 1.0, 0.0], dtype=float)
    right = np.cross(forward, up_vector)
    right /= np.linalg.norm(right)

    if keys[pygame.K_w]: cam_pos += forward * movement_speed * dt
    if keys[pygame.K_s]: cam_pos -= forward * movement_speed * dt
    if keys[pygame.K_a]: cam_pos -= right * movement_speed * dt
    if keys[pygame.K_d]: cam_pos += right * movement_speed * dt
    if keys[pygame.K_SPACE]: cam_pos += up_vector * movement_speed * dt # Use up_vector for vertical movement
    if keys[pygame.K_LSHIFT]: cam_pos -= up_vector * movement_speed * dt # Use up_vector for vertical movement

    mouse_dx, mouse_dy = pygame.mouse.get_rel()
    cam_yaw += mouse_dx * mouse_sensitivity
    cam_pitch -= mouse_dy * mouse_sensitivity
    cam_pitch = max(-89.0, min(89.0, cam_pitch))

    wp.launch(kernel=update_particles, dim=num_particles, inputs=[positions_gpu, velocities_gpu, float(dt), float(cube_size), float(screen_width), float(screen_height), float(screen_depth)], device="cuda")
    wp.copy(positions_gpu, positions_cpu)
    positions_np = positions_cpu.numpy()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    apply_camera_transform()

    draw_bounds(screen_width, screen_height, screen_depth)

    for i in range(num_particles):
        pos = positions_np[i]
        glPushMatrix()
        glTranslatef(*pos)
        draw_cube(cube_size)
        glPopMatrix()

    pygame.display.flip()

pygame.quit()