import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import numpy as np
import warp as wp
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import glutInit, glutSolidSphere

# --- Simulation Parameters ---
num_particles = 128
screen_width, screen_height, screen_depth = 800, 600, 600
radius = 12.0

# --- Warp Physics Kernel, fully float-safe and robust ---
@wp.kernel
def update_particles(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
    radius: float,
    bound_x: float,
    bound_y: float,
    bound_z: float
):
    tid = wp.tid()
    pos = positions[tid]
    vel = velocities[tid]

    # Gravity
    vel += wp.vec3(0.0, -9.8 * dt, 0.0)
    pos += vel * dt

    # Elastic collisions (O(N^2), suitable for < 1000 particles)
    for j in range(positions.shape[0]):
        if j != tid:
            other_pos = positions[j]
            other_vel = velocities[j]
            delta = pos - other_pos
            dist = wp.length(delta)
            min_dist = 2.0 * radius  # float32!
            if dist < min_dist and dist > 1e-6:
                normal = delta / dist
                rel_vel = vel - other_vel
                sep_vel = wp.dot(rel_vel, normal)
                if sep_vel < 0.0:  # Only adjust if approaching
                    # Exchange velocities along collision axis (equal mass)
                    vel = vel - sep_vel * normal
                    # Push apart to avoid overlap
                    overlap = min_dist - dist
                    pos = pos + 0.5 * overlap * normal

    # Wall boundary bounce (float32 math everywhere!)
    if pos[0] < radius or pos[0] > bound_x - radius:
        vel = wp.vec3(-vel[0], vel[1], vel[2])
        pos = wp.vec3(
            wp.clamp(pos[0], radius, bound_x-radius),
            wp.clamp(pos[1], radius, bound_y-radius),
            wp.clamp(pos[2], radius, bound_z-radius)
        )
    if pos[1] < radius or pos[1] > bound_y - radius:
        vel = wp.vec3(vel[0], -vel[1], vel[2])
        pos = wp.vec3(
            wp.clamp(pos[0], radius, bound_x-radius),
            wp.clamp(pos[1], radius, bound_y-radius),
            wp.clamp(pos[2], radius, bound_z-radius)
        )
    if pos[2] < radius or pos[2] > bound_z - radius:
        vel = wp.vec3(vel[0], vel[1], -vel[2])
        pos = wp.vec3(
            wp.clamp(pos[0], radius, bound_x-radius),
            wp.clamp(pos[1], radius, bound_y-radius),
            wp.clamp(pos[2], radius, bound_z-radius)
        )

    positions[tid] = pos
    velocities[tid] = vel

# --- Initialization ---
wp.init()
glutInit()  # REQUIRED before using glutSolidSphere!

# Positions: uniformly inside box, keeping 1 radius distance from walls
init_positions = np.random.rand(num_particles, 3).astype(np.float32)
init_positions *= np.array([screen_width, screen_height, screen_depth], dtype=np.float32) - 2.0 * radius
init_positions += radius

positions_gpu = wp.from_numpy(init_positions, dtype=wp.vec3, device="cuda")
init_vel = (np.random.rand(num_particles, 3)-0.5) * 120.0
velocities_gpu = wp.from_numpy(init_vel.astype(np.float32), dtype=wp.vec3, device="cuda")
positions_cpu = wp.zeros(num_particles, dtype=wp.vec3, device="cpu")

pygame.init()
pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Warp 3D Spheres: Gravity & Elastic Collisions")

glEnable(GL_DEPTH_TEST)
gluPerspective(45.0, float(screen_width) / float(screen_height), 0.1, 2000.0)

cam_x, cam_y, cam_z = screen_width/2.0, screen_height/2.0, -900.0
cam_look = [screen_width/2.0, screen_height/2.0, screen_depth/2.0]

clock = pygame.time.Clock()
running = True

while running:
    dt = clock.tick(60) / 1000.0  # float
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Camera movement controls (optional, e.g. WASD):

    # Physics step - all kernel args are float!
    wp.launch(
        kernel=update_particles,
        dim=num_particles,
        inputs=[
            positions_gpu, velocities_gpu,
            float(dt), float(radius),
            float(screen_width), float(screen_height), float(screen_depth)
        ],
        device="cuda"
    )
    wp.copy(positions_cpu, positions_gpu)
    positions = positions_cpu.numpy()

    # Rendering - safe GLUT usage
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(cam_x, cam_y, cam_z, cam_look[0], cam_look[1], cam_look[2], 0.0, 1.0, 0.0)

    for x, y, z in positions:
        glPushMatrix()
        glTranslatef(float(x), float(y), float(z))
        glColor3f(0.2, 0.8, 0.4)
        glutSolidSphere(float(radius), 12, 12)
        glPopMatrix()

    pygame.display.flip()

pygame.quit()