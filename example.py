"""
CPU Rigid-Body Physics Demo: rotating boxes (OBBs) and rotating spheres

- Boxes are full rigid bodies (position, orientation quaternion, linear and angular velocity).
- Spheres are spheres with angular velocity (spin) and rotational inertia.
- Collision types: sphere-sphere, sphere-box (OBB), box-box (OBB vs OBB) using SAT (separating axis theorem).
- Collision response: impulse-based including angular response, mass & inertia considered.
- Positional correction to reduce sinking.
- Simple rendering with PyOpenGL and pygame. Boxes render with orientation applied.
- Uses fixed substepping for stable simulation.

Requirements:
    pip install pygame PyOpenGL numpy

Notes:
- This is still a simplified engine (boxes are non-deforming, no friction, no stacking solver).
- OBB vs OBB SAT is used; it's not as robust as a production solver but works for demo purposes.
"""
import sys
import math
import random
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *

# Window / scene settings
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 700
FOV = 60.0
NEAR = 0.1
FAR = 2000.0

# Physics settings
GRAVITY = np.array([0.0, -9.81, 0.0], dtype=float)
RESTIUTION = 0.6            # coefficient of restitution for collisions
POSITION_CORRECTION = 0.8   # positional correction factor (0..1)
PENETRATION_SLOP = 0.01     # tolerance before applying positional correction
DT_FIXED = 1.0 / 120.0      # fixed physics sub-step

# Simulation bounds (AABB)
BOUNDS_MIN = np.array([-50.0, 0.0, -50.0], dtype=float)
BOUNDS_MAX = np.array([ 50.0, 40.0, 50.0], dtype=float)

# Objects
NUM_OBJECTS = 18

# Simple density for mass computation
DENSITY = 1.0

# Rendering detail
SPHERE_SLICES = 20
SPHERE_STACKS = 14

# Object list
objects = []

# ----------------------
# Quaternion utilities
# ----------------------
def quat_mul(q1, q2):
    # q = [w, x, y, z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_from_axis_angle(axis, angle):
    axis = np.array(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / n
    s = math.sin(angle * 0.5)
    return np.array([math.cos(angle * 0.5), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)

def quat_to_matrix(q):
    w,x,y,z = q
    # 3x3 rotation matrix
    return np.array([
        [1 - 2*y*y - 2*z*z,   2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,       1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,       2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ], dtype=float)

# ----------------------
# Inertia calculators
# ----------------------
def box_inertia_tensor(size, mass):
    # size = (w,h,d)
    w,h,d = size
    Ixx = (1.0/12.0) * mass * (h*h + d*d)
    Iyy = (1.0/12.0) * mass * (w*w + d*d)
    Izz = (1.0/12.0) * mass * (w*w + h*h)
    return np.diag([Ixx, Iyy, Izz])

def sphere_inertia_tensor(radius, mass):
    I = (2.0/5.0) * mass * (radius**2)
    return np.eye(3) * I

# ----------------------
# Object creation
# ----------------------
def random_box_size():
    return (random.uniform(1.2, 4.0), random.uniform(1.0, 6.0), random.uniform(1.2, 4.0))

def random_sphere_radius():
    return random.uniform(0.6, 2.2)

def spawn_objects():
    for i in range(NUM_OBJECTS):
        if random.random() < 0.6:
            # box
            size = random_box_size()
            vol = size[0]*size[1]*size[2]
            mass = max(0.0001, DENSITY * vol)
            pos = np.array([
                random.uniform(BOUNDS_MIN[0]+6.0, BOUNDS_MAX[0]-6.0),
                random.uniform(BOUNDS_MIN[1]+6.0, BOUNDS_MAX[1]-6.0),
                random.uniform(BOUNDS_MIN[2]+6.0, BOUNDS_MAX[2]-6.0)
            ], dtype=float)
            vel = np.array([random.uniform(-6,6), random.uniform(-2,2), random.uniform(-6,6)], dtype=float)
            # small random orientation
            axis = np.random.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            angle = random.uniform(0, math.pi*2)
            q = quat_from_axis_angle(axis, angle)
            # angular velocity
            omega = np.array([random.uniform(-2,2), random.uniform(-2,2), random.uniform(-2,2)], dtype=float)
            I_body = box_inertia_tensor(size, mass)
            I_body_inv = np.linalg.inv(I_body)
            objects.append({
                'type':'box',
                'size': np.array(size, dtype=float),
                'pos': pos,
                'vel': vel,
                'orientation': q,
                'omega': omega,
                'mass': mass,
                'inv_mass': 1.0/mass,
                'I_body': I_body,
                'I_body_inv': I_body_inv,
                'restitution': RESTIUTION
            })
        else:
            # sphere
            r = random_sphere_radius()
            vol = (4.0/3.0) * math.pi * r**3
            mass = max(0.0001, DENSITY * vol)
            pos = np.array([
                random.uniform(BOUNDS_MIN[0]+6.0, BOUNDS_MAX[0]-6.0),
                random.uniform(BOUNDS_MIN[1]+6.0, BOUNDS_MAX[1]-6.0),
                random.uniform(BOUNDS_MIN[2]+6.0, BOUNDS_MAX[2]-6.0)
            ], dtype=float)
            vel = np.array([random.uniform(-6,6), random.uniform(-6,6), random.uniform(-6,6)], dtype=float)
            # orientation quaternion for sphere (visual spin)
            axis = np.random.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-12)
            angle = random.uniform(0, math.pi*2)
            q = quat_from_axis_angle(axis, angle)
            omega = np.array([random.uniform(-6,6), random.uniform(-6,6), random.uniform(-6,6)], dtype=float)
            I_body = sphere_inertia_tensor(r, mass)
            I_body_inv = np.linalg.inv(I_body)
            objects.append({
                'type':'sphere',
                'radius': r,
                'pos': pos,
                'vel': vel,
                'orientation': q,
                'omega': omega,
                'mass': mass,
                'inv_mass': 1.0/mass,
                'I_body': I_body,
                'I_body_inv': I_body_inv,
                'restitution': RESTIUTION
            })

# ----------------------
# Geometry helpers
# ----------------------
def get_box_corners(obj):
    # returns 8 world-space corner positions of the OBB
    w,h,d = obj['size']
    hx,hy,hz = 0.5*w, 0.5*h, 0.5*d
    local = np.array([
        [ hx,  hy,  hz],
        [ hx, -hy,  hz],
        [-hx, -hy,  hz],
        [-hx,  hy,  hz],
        [ hx,  hy, -hz],
        [ hx, -hy, -hz],
        [-hx, -hy, -hz],
        [-hx,  hy, -hz],
    ], dtype=float).T  # 3x8
    R = quat_to_matrix(obj['orientation'])
    world = (R @ local).T + obj['pos']
    return world  # shape (8,3)

def project_points_on_axis(points, axis):
    axis_n = axis / (np.linalg.norm(axis) + 1e-12)
    projs = points @ axis_n
    return np.min(projs), np.max(projs)

# Point in OBB test (world space)
def point_in_obb(point, obj):
    R = quat_to_matrix(obj['orientation'])
    local = R.T @ (point - obj['pos'])
    half = 0.5 * obj['size']
    return np.all(local >= -half - 1e-8) and np.all(local <= half + 1e-8)

# ----------------------
# Collision detection
# ----------------------
def sphere_sphere_collision(a, b):
    diff = b['pos'] - a['pos']
    dist2 = np.dot(diff, diff)
    rsum = a['radius'] + b['radius']
    if dist2 >= (rsum * rsum):
        return False, None, 0.0, None
    dist = math.sqrt(dist2) if dist2 > 1e-12 else 0.0
    if dist > 0.0:
        normal = diff / dist
        contact_point = a['pos'] + normal * a['radius']
    else:
        normal = np.array([1.0,0.0,0.0], dtype=float)
        contact_point = a['pos']
    penetration = rsum - dist
    return True, normal, penetration, contact_point

def sphere_box_collision(s, b):
    bmin = get_box_corners(b).min(axis=0)
    bmax = get_box_corners(b).max(axis=0)
    # compute closest point on OBB to sphere center by using OBB local coords
    R = quat_to_matrix(b['orientation'])
    local_center = R.T @ (s['pos'] - b['pos'])
    half = 0.5 * b['size']
    closest_local = np.clip(local_center, -half, half)
    closest_world = b['pos'] + (R @ closest_local)
    diff = s['pos'] - closest_world
    dist2 = np.dot(diff, diff)
    if dist2 >= (s['radius'] * s['radius']):
        return False, None, 0.0, None
    dist = math.sqrt(dist2) if dist2 > 1e-12 else 0.0
    if dist > 1e-12:
        normal = diff / dist
    else:
        # sphere center inside box: pick nearest face normal in local coords
        dists = np.array([
            half[0] - abs(local_center[0]),
            half[1] - abs(local_center[1]),
            half[2] - abs(local_center[2]),
        ])
        idx = np.argmin(dists)
        n_local = np.zeros(3)
        n_local[idx] = 1.0 if local_center[idx] >= 0 else -1.0
        normal = (R @ n_local)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
    penetration = s['radius'] - dist
    contact_point = closest_world
    return True, normal, penetration, contact_point

def obb_obb_collision(a, b):
    # SAT using corners projection: collect candidate axes = face normals of both OBBs and cross products of edges
    a_corners = get_box_corners(a)
    b_corners = get_box_corners(b)

    # get local axes (world-space) as columns of rotation matrix
    A = quat_to_matrix(a['orientation'])
    B = quat_to_matrix(b['orientation'])
    axes = []
    for i in range(3):
        axes.append(A[:,i])
    for i in range(3):
        axes.append(B[:,i])
    # cross products
    for i in range(3):
        for j in range(3):
            cp = np.cross(A[:,i], B[:,j])
            if np.linalg.norm(cp) > 1e-6:
                axes.append(cp)

    min_pen = float('inf')
    min_axis = None
    for axis in axes:
        axis_n = axis / (np.linalg.norm(axis) + 1e-12)
        a_min, a_max = project_points_on_axis(a_corners, axis_n)
        b_min, b_max = project_points_on_axis(b_corners, axis_n)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap <= 0:
            return False, None, 0.0, None
        if overlap < min_pen:
            min_pen = overlap
            # normal should point from a to b
            center_dir = b['pos'] - a['pos']
            if np.dot(center_dir, axis_n) < 0:
                axis_n = -axis_n
            min_axis = axis_n

    # approximate contact point as midpoint of centers pushed along normal
    contact_point = (a['pos'] + b['pos']) * 0.5 - min_axis * (min_pen * 0.5)
    return True, min_axis, min_pen, contact_point

# ----------------------
# Collision resolution
# ----------------------
def apply_impulse(a, b, contact_point, normal, penetration):
    # r vectors from centers to contact
    ra = contact_point - a['pos']
    rb = contact_point - b['pos']

    # relative velocity at contact
    va = a['vel'] + np.cross(a['omega'], ra)
    vb = b['vel'] + np.cross(b['omega'], rb)
    rel = vb - va
    vel_along_normal = np.dot(rel, normal)
    if vel_along_normal > 0:
        return

    # compute inv inertia in world space
    Ra = quat_to_matrix(a['orientation'])
    Rb = quat_to_matrix(b['orientation'])
    Ia_inv = Ra @ a['I_body_inv'] @ Ra.T
    Ib_inv = Rb @ b['I_body_inv'] @ Rb.T

    inv_mass_sum = a['inv_mass'] + b['inv_mass']
    # rotational part
    ra_x_n = np.cross(ra, normal)
    rb_x_n = np.cross(rb, normal)
    rot_term = np.dot(normal, np.cross(Ia_inv @ ra_x_n, ra) + np.cross(Ib_inv @ rb_x_n, rb))
    denom = inv_mass_sum + rot_term
    if denom <= 1e-12:
        return

    e = min(a.get('restitution', RESTIUTION), b.get('restitution', RESTIUTION))
    j = -(1 + e) * vel_along_normal / denom
    impulse = j * normal

    # apply linear
    a['vel'] -= impulse * a['inv_mass']
    b['vel'] += impulse * b['inv_mass']
    # apply angular
    a['omega'] -= Ia_inv @ np.cross(ra, impulse)
    b['omega'] += Ib_inv @ np.cross(rb, impulse)

    # positional correction (Baumgarte-style)
    correction_mag = max(penetration - PENETRATION_SLOP, 0.0) / (a['inv_mass'] + b['inv_mass'])
    correction = correction_mag * POSITION_CORRECTION * normal
    a['pos'] -= correction * a['inv_mass']
    b['pos'] += correction * b['inv_mass']

# ----------------------
# Physics integrator
# ----------------------
def physics_step(dt):
    # integrate linear and angular
    for obj in objects:
        if obj['inv_mass'] == 0.0:
            continue
        # linear semi-implicit Euler
        obj['vel'] += GRAVITY * dt
        obj['pos'] += obj['vel'] * dt
        # angular: integrate omega (no torques here except impulses)
        # integrate orientation via quaternion derivative: q_dot = 0.5 * quat(w=0, omega) * q
        omega = obj['omega']
        q = obj['orientation']
        q_omega = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
        q_dot = 0.5 * quat_mul(q_omega, q)
        q_new = q + q_dot * dt
        obj['orientation'] = quat_normalize(q_new)
        # simple angular damping for stability
        obj['omega'] *= 0.999

    # broadphase naive: pairwise
    n = len(objects)
    for i in range(n):
        for j in range(i+1, n):
            A = objects[i]
            B = objects[j]
            colliding = False
            normal = None
            penetration = 0.0
            contact_point = None
            if A['type'] == 'sphere' and B['type'] == 'sphere':
                colliding, normal, penetration, contact_point = sphere_sphere_collision(A,B)
            elif A['type'] == 'sphere' and B['type'] == 'box':
                colliding, normal, penetration, contact_point = sphere_box_collision(A,B)
            elif A['type'] == 'box' and B['type'] == 'sphere':
                colliding, normal, penetration, contact_point = sphere_box_collision(B,A)
                if colliding:
                    normal = -normal  # we computed normal from sphere to box
            else:
                colliding, normal, penetration, contact_point = obb_obb_collision(A,B)

            if colliding and contact_point is not None:
                apply_impulse(A, B, contact_point, normal, penetration)

    # walls collision for each object (approximate using object's bounding radius/half extents)
    for obj in objects:
        if obj['type'] == 'sphere':
            half = np.array([obj['radius'], obj['radius'], obj['radius']])
        else:
            half = 0.5 * obj['size']
        min_corner = obj['pos'] - half
        max_corner = obj['pos'] + half
        # X
        if min_corner[0] < BOUNDS_MIN[0]:
            obj['pos'][0] = BOUNDS_MIN[0] + half[0]
            obj['vel'][0] = -obj['vel'][0] * obj.get('restitution', RESTIUTION)
            obj['omega'] *= 0.9
        if max_corner[0] > BOUNDS_MAX[0]:
            obj['pos'][0] = BOUNDS_MAX[0] - half[0]
            obj['vel'][0] = -obj['vel'][0] * obj.get('restitution', RESTIUTION)
            obj['omega'] *= 0.9
        # Y
        if min_corner[1] < BOUNDS_MIN[1]:
            obj['pos'][1] = BOUNDS_MIN[1] + half[1]
            obj['vel'][1] = -obj['vel'][1] * obj.get('restitution', RESTIUTION)
            obj['omega'] *= 0.9
        if max_corner[1] > BOUNDS_MAX[1]:
            obj['pos'][1] = BOUNDS_MAX[1] - half[1]
            obj['vel'][1] = -obj['vel'][1] * obj.get('restitution', RESTIUTION)
            obj['omega'] *= 0.9
        # Z
        if min_corner[2] < BOUNDS_MIN[2]:
            obj['pos'][2] = BOUNDS_MIN[2] + half[2]
            obj['vel'][2] = -obj['vel'][2] * obj.get('restitution', RESTIUTION)
            obj['omega'] *= 0.9
        if max_corner[2] > BOUNDS_MAX[2]:
            obj['pos'][2] = BOUNDS_MAX[2] - half[2]
            obj['vel'][2] = -obj['vel'][2] * obj.get('restitution', RESTIUTION)
            obj['omega'] *= 0.9

# ----------------------
# Rendering helpers
# ----------------------
quadric = None
def draw_sphere(radius):
    global quadric
    if quadric is None:
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
    glColor3f(0.8, 0.25, 0.25)
    gluSphere(quadric, radius, SPHERE_SLICES, SPHERE_STACKS)

def draw_box(size):
    w,h,d = size
    hw,hh,hd = w*0.5,h*0.5,d*0.5
    vertices = [
        ( hw,  hh,  hd), ( hw, -hh,  hd), (-hw, -hh,  hd), (-hw,  hh,  hd),
        ( hw,  hh, -hd), ( hw, -hh, -hd), (-hw, -hh, -hd), (-hw,  hh, -hd)
    ]
    faces = [(0,1,2,3),(4,5,6,7),(0,4,5,1),(3,2,6,7),(0,3,7,4),(1,5,6,2)]
    glBegin(GL_QUADS)
    glColor3f(0.2, 0.8, 0.4)
    for face in faces:
        for idx in face:
            glVertex3f(*vertices[idx])
    glEnd()
    # edges
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    glColor3f(0.03,0.03,0.03)
    glBegin(GL_LINES)
    for a,b in edges:
        glVertex3f(*vertices[a])
        glVertex3f(*vertices[b])
    glEnd()

def draw_grid(size=120, step=5):
    glDisable(GL_LIGHTING)
    glColor3f(0.35, 0.35, 0.35)
    glBegin(GL_LINES)
    half = size * 0.5
    for i in range(-int(half), int(half)+1, step):
        glVertex3f(-half, 0.0, i)
        glVertex3f(half, 0.0, i)
        glVertex3f(i, 0.0, -half)
        glVertex3f(i, 0.0, half)
    glEnd()

def draw_bounds():
    x0,x1 = BOUNDS_MIN[0], BOUNDS_MAX[0]
    y0,y1 = BOUNDS_MIN[1], BOUNDS_MAX[1]
    z0,z1 = BOUNDS_MIN[2], BOUNDS_MAX[2]
    corners = [
        (x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1),
        (x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)
    ]
    lines = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    glColor3f(0.2, 0.6, 0.9)
    glBegin(GL_LINES)
    for a,b in lines:
        glVertex3f(*corners[a])
        glVertex3f(*corners[b])
    glEnd()

# Utility to push transform from position & quaternion
def gl_push_transform(pos, quat):
    R = quat_to_matrix(quat)
    # Build 4x4 column-major matrix for OpenGL
    mat = np.eye(4, dtype=float)
    mat[:3,:3] = R
    mat[:3,3] = pos
    glPushMatrix()
    glMultMatrixf(mat.T)  # OpenGL expects column-major; numpy is row-major, so transpose

# ----------------------
# OpenGL init
# ----------------------
def init_opengl():
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glClearColor(0.12, 0.12, 0.12, 1.0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)

def resize(w,h):
    if h == 0:
        h = 1
    glViewport(0,0,w,h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(FOV, w/float(h), NEAR, FAR)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

# ----------------------
# Main
# ----------------------
def main():
    pygame.init()
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("CPU Rigid Bodies: Rotating Boxes & Spheres")
    init_opengl()
    resize(SCREEN_WIDTH, SCREEN_HEIGHT)

    cam_distance = 110.0
    cam_pitch = 18.0
    cam_yaw = -45.0

    spawn_objects()

    clock = pygame.time.Clock()
    running = True

    while running:
        frame_dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            cam_yaw -= 50.0 * frame_dt
        if keys[pygame.K_RIGHT]:
            cam_yaw += 50.0 * frame_dt
        if keys[pygame.K_UP]:
            cam_pitch = min(89.0, cam_pitch + 50.0 * frame_dt)
        if keys[pygame.K_DOWN]:
            cam_pitch = max(-20.0, cam_pitch - 50.0 * frame_dt)

        # fixed-step physics
        t = frame_dt
        while t > 0.0:
            step = min(DT_FIXED, t)
            physics_step(step)
            t -= step

        # render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        yaw_rad = math.radians(cam_yaw)
        pitch_rad = math.radians(cam_pitch)
        cx = cam_distance * math.cos(pitch_rad) * math.sin(yaw_rad)
        cy = cam_distance * math.sin(pitch_rad)
        cz = cam_distance * math.cos(pitch_rad) * math.cos(yaw_rad)
        eye = np.array([cx, cy + 12.0, cz], dtype=float)
        center = np.array([0.0, 6.0, 0.0], dtype=float)
        up = np.array([0.0, 1.0, 0.0], dtype=float)
        gluLookAt(eye[0], eye[1], eye[2],
                  center[0], center[1], center[2],
                  up[0], up[1], up[2])

        draw_grid(size=140, step=5)
        draw_bounds()

        for obj in objects:
            gl_push_transform(obj['pos'], obj['orientation'])
            if obj['type'] == 'box':
                draw_box(obj['size'])
            else:
                # rotate sphere visually according to orientation
                glPushMatrix()
                # orientation already applied by gl_push_transform, just draw sphere at origin
                draw_sphere(obj['radius'])
                glPopMatrix()
            glPopMatrix()

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
