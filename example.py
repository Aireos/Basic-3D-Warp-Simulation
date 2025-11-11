import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import numpy as np
import warp as wp
from OpenGL.GL import *
from OpenGL.GLU import *

# --- Simulation Parameters ---
num_particles = 128
screen_width, screen_height, screen_depth = 800, 600, 600
cube_size = 24.0

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

    # Gravity
    vel += wp.vec3(0.0, -9.8 * dt, 0.0)
    pos += vel * dt

    # Elastic collisions as axis-aligned cubes (naive O(N^2), suitable for < 1000 particles)
    # This checks if two cubes overlap and adjusts velocity roughly
    half = 0.5 * cube_size
    for j in range(positions.shape[0]):
        if j != tid:
            other_pos = positions[j]
            dx = abs(pos[0] - other_pos[0])
            dy = abs(pos[1] - other_pos[1])
            dz = abs(pos[2] - other_pos[2])
            overlap = (dx < cube_size) and (dy < cube_size) and (dz < cube_size)
            if overlap:
                # Resolve minimal axis of separation for cubes
                if dx < dy and dx < dz:
                    sign = 1.0 if pos[0] > other_pos[0] else -1.0
                    vel = wp.vec3(sign * abs(vel[0]), vel[1], vel[2])
                    pos = pos + wp.vec3(sign * (cube_size - dx) * 0.5, 0.0, 0.0)
                elif dy < dz:
                    sign = 1.0 if pos[1] > other_pos[1] else -1.0
                    vel = wp.vec3(vel[0], sign * abs(vel[1]), vel[2])
                    pos = pos + wp.vec3(0.0, sign * (cube_size - dy) * 0.5, 0.0)
                else:
                    sign = 1.0 if pos[2] > other_pos[2] else -1.0
                    vel = wp.vec3(vel[0], vel[1], sign * abs(vel[2]))
                    pos = pos + wp.vec3(0.0, 0.0, sign * (cube_size - dz) * 0.5)

    # Wall boundary bounce
    if pos[0] < half or pos[0] > bound_x - half:
        vel = wp.vec3(-vel[0], vel[1], vel[2])
        pos = wp.vec3(
            wp.clamp(pos[0], half, bound_x-half),
            wp.clamp(pos[1], half, bound_y-half),
            wp.clamp(pos[2], half, bound_z-half)
        )
    if pos[1] < half or pos[1] > bound_y - half:
        vel = wp.vec3(vel[0], -vel[1], vel[2])
        pos = wp.vec3(
            wp.clamp(pos[0], half, bound_x-half),
            wp.clamp(pos[1], half, bound_y-half),
            wp.clamp(pos[2], half, bound_z-half)
        )
    if pos[2] < half or pos[2] > bound_z - half:
        vel = wp.vec3(vel[0], vel[1], -vel[2])
        pos = wp.vec3(
            wp.clamp(pos[0], half, bound_x-half),
            wp.clamp(pos[1], half, bound_y-half),
            wp.clamp(pos[2], half, bound_z-half)
        )

    positions[tid] = pos
    velocities[tid] = vel

# --- Initialization ---
wp.init()
# No GLUT needed—using custom cube rendering!
init_positions = np.random.rand(num_particles, 3).astype(np.float32)
init_positions *= np.array([screen_width, screen_height, screen_depth], dtype=np.float32) - cube_size
init_positions += cube_size * 0.5
positions_gpu = wp.from_numpy(init_positions, dtype=wp.vec3, device="cuda")
init_vel = (np.random.rand(num_particles, 3) - 0.5) * 120.0
velocities_gpu = wp.from_numpy(init_vel.astype(np.float32), dtype=wp.vec3, device="cuda")
positions_cpu = wp.zeros(num_particles, dtype=wp.vec3, device="cpu")

pygame.init()
pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Warp 3D Cubes: Gravity & Elastic Collisions")

glEnable(GL_DEPTH_TEST)
gluPerspective(45.0, float(screen_width) / float(screen_height), 0.1, 2000.0)

cam_x, cam_y, cam_z = screen_width/2.0, screen_height/2.0, -900.0
cam_look = [screen_width/2.0, screen_height/2.0, screen_depth/2.0]
clock = pygame.time.Clock()
running = True

def draw_cube(cube_size):
    half = cube_size * 0.5
    vertices = [
        [ half,  half,  half], [ half, -half,  half], [-half, -half,  half], [-half,  half,  half], # Front face
        [ half,  half, -half], [ half, -half, -half], [-half, -half, -half], [-half,  half, -half]  # Back face
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0), # Front square
        (4,5), (5,6), (6,7), (7,4), # Back square
        (0,4), (1,5), (2,6), (3,7)  # Side edges
    ]
    faces = [
        (0,1,2,3),      # Front
        (7,6,5,4),      # Back
        (0,4,5,1),      # Right
        (3,2,6,7),      # Left
        (0,3,7,4),      # Top
        (1,5,6,2)       # Bottom
    ]
    # Draw faces (solid colored cube)
    glBegin(GL_QUADS)
    for idx, face in enumerate(faces):
        glColor3f(0.2, 0.8, 0.4)  # Greenish color
        for vertex in face:
            glVertex3f(*vertices[vertex])
    glEnd()
    # Draw edges
    glColor3f(0,0,0)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3f(*vertices[vertex])
    glEnd()

while running:
    dt = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    # (Add WASD camera movement here if desired)
    wp.launch(
        kernel=update_particles,
        dim=num_particles,
        inputs=[
            positions_gpu, velocities_gpu,
            float(dt), float(cube_size),
            float(screen_width), float(screen_height), float(screen_depth)
        ],
        device="cuda"
    )
    wp.copy(positions_cpu, positions_gpu)
    positions = positions_cpu.numpy()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(cam_x, cam_y, cam_z, cam_look[0], cam_look[1], cam_look[2], 0.0, 1.0, 0.0)

    for x, y, z in positions:
        glPushMatrix()
        glTranslatef(float(x), float(y), float(z))
        draw_cube(cube_size)
        glPopMatrix()

    pygame.display.flip()
pygame.quit()