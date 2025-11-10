import warp as wp
import pygame
import numpy as np

# --- 1. Warp Simulation Setup ---
wp.init()

num_particles = 1024
screen_width, screen_height = 800, 600

@wp.kernel
def update_particles(
    positions: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    dt: float,
    screen_w: int,
    screen_h: int
):
    tid = wp.tid()
    pos = positions[tid]
    vel = velocities[tid]

    # Apply gravity (simple downward force)
    vel += wp.vec2(0.0, 9.8 * dt)

    # Update position
    pos += vel * dt

    # Boundary conditions (bounce off edges)
    if pos[0] < 0.0 or pos[0] > screen_w:
        vel *= wp.vec2(-1.0, 1.0)
        pos = wp.clamp(pos, wp.vec2(0.0, 0.0), wp.vec2(screen_w, screen_h)) # Clamp to stay in bounds
    if pos[1] < 0.0 or pos[1] > screen_h:
        vel *= wp.vec2(1.0, -1.0)
        pos = wp.clamp(pos, wp.vec2(0.0, 0.0), wp.vec2(screen_w, screen_h))

    positions[tid] = pos
    velocities[tid] = vel

# Initialize particle positions and velocities on the GPU
positions_gpu = wp.zeros(num_particles, dtype=wp.vec2, device="cuda")
velocities_gpu = wp.zeros(num_particles, dtype=wp.vec2, device="cuda")

# Initial positions (randomized)
# Copy initial random positions from CPU (NumPy) to GPU (Warp array)
initial_positions_cpu = np.random.rand(num_particles, 2).astype(np.float32) * [screen_width, screen_height]
wp.copy(positions_gpu, initial_positions_cpu)

# --- 2. Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("NVIDIA Warp Simulation with Pygame")
clock = pygame.time.Clock()
running = True

# Buffer for copying data from GPU to CPU
positions_cpu = np.zeros((num_particles, 2), dtype=np.float32)

# --- 3. Main Game Loop ---
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Simulation Step (Warp) ---
    dt = clock.tick(60) / 1000.0  # Time delta in seconds, capped at 60 FPS
    wp.launch(
        kernel=update_particles,
        dim=num_particles,
        inputs=[positions_gpu, velocities_gpu, dt, screen_width, screen_height],
        device="cuda"
    )

    # --- Data Transfer (GPU to CPU) ---
    # Asynchronously copy the positions data to the CPU buffer
    wp.copy(positions_cpu, positions_gpu)

    # --- Rendering Step (Pygame) ---
    screen.fill((0, 0, 0))  # Clear screen with black
    for i in range(num_particles):
        x, y = positions_cpu[i]
        pygame.draw.circle(screen, (0, 255, 0), (int(x), int(y)), 5) # Draw particles as green circles

    pygame.display.flip()  # Update the display

pygame.quit()
