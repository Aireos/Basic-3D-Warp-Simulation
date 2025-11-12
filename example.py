"""
CPU Rigid-Body Physics Demo: rotating boxes (OBBs) and rotating spheres
Optimized rewrite:
- Per-substep caching: rotation matrix, world inertia inverse, corners, AABB
- Simple AABB broadphase to cull pairs
- Reduced redundant numpy allocations
- Slightly reduced rendering tessellation
- Fixed-step physics with max substeps per frame

Requirements:
    pip install pygame PyOpenGL numpy
"""

import sys
import math
import random
import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *

# ----------------------
# Config
# ----------------------
SCREEN_WIDTH = 1100
SCREEN_HEIGHT = 700
FOV = 60.0
NEAR = 0.1
FAR = 2000.0

GRAVITY = np.array([0.0, -9.81, 0.0], dtype=float)
RESTITUTION_DEFAULT = 0.6
POSITION_CORRECTION = 0.8
PENETRATION_SLOP = 0.01
DT_FIXED = 1.0 / 90.0          # try 1/60..1/120
MAX_SUBSTEPS_PER_FRAME = 3     # safety to avoid spiral of death

BOUNDS_MIN = np.array([-50.0, 0.0, -50.0], dtype=float)
BOUNDS_MAX = np.array([ 50.0, 40.0, 50.0], dtype=float)

NUM_OBJECTS = 18
DENSITY = 1.0

# Rendering detail (reduced a bit)
SPHERE_SLICES = 12
SPHERE_STACKS = 8
GRID_STEP = 10

# ----------------------
# Globals
# ----------------------
objects = []
quadric = None

# ----------------------
# Math utils (lean)
# ----------------------
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)

def quat_normalize_inplace(q):
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    else:
        q[:] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q

def quat_from_axis_angle(axis, angle):
    ax = np.asarray(axis, dtype=float)
    n = np.linalg.norm(ax)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    ax /= n
    s = math.sin(angle * 0.5)
    return np.array([math.cos(angle * 0.5), ax[0]*s, ax[1]*s, ax[2]*s], dtype=float)

def quat_to_matrix(q):
    w, x, y, z = q
    xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ], dtype=float)

def clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

# ----------------------
# Inertia
# ----------------------
def box_inertia_tensor(size, mass):
    w,h,d = size
    Ixx = (1.0/12.0) * mass * (h*h + d*d)
    Iyy = (1.0/12.0) * mass * (w*w + d*d)
    Izz = (1.0/12.0) * mass * (w*w + h*h)
    return np.diag([Ixx, Iyy, Izz])

def sphere_inertia_tensor(radius, mass):
    I = (2.0/5.0) * mass * (radius**2)
    return np.eye(3) * I

# ----------------------
# Spawn
# ----------------------
def random_box_size():
    return np.array([
        random.uniform(1.2, 4.0),
        random.uniform(1.0, 6.0),
        random.uniform(1.2, 4.0)
    ], dtype=float)

def random_sphere_radius():
    return random.uniform(0.6, 2.2)

def spawn_objects():
    objects.clear()
    for _ in range(NUM_OBJECTS):
        if random.random() < 0.6:
            size = random_box_size()
            vol = size[0]*size[1]*size[2]
            mass = max(0.0001, DENSITY * vol)
            pos = np.array([
                random.uniform(BOUNDS_MIN[0]+6.0, BOUNDS_MAX[0]-6.0),
                random.uniform(BOUNDS_MIN[1]+6.0, BOUNDS_MAX[1]-6.0),
                random.uniform(BOUNDS_MIN[2]+6.0, BOUNDS_MAX[2]-6.0)
            ], dtype=float)
            vel = np.array([random.uniform(-6,6), random.uniform(-2,2), random.uniform(-6,6)], dtype=float)
            axis = np.random.normal(size=3)
            n = np.linalg.norm(axis)
            axis = axis / (n + 1e-12)
            angle = random.uniform(0, math.pi*2)
            q = quat_from_axis_angle(axis, angle)
            omega = np.array([random.uniform(-2,2), random.uniform(-2,2), random.uniform(-2,2)], dtype=float)
            I_body = box_inertia_tensor(size, mass)
            I_body_inv = np.linalg.inv(I_body)
            objects.append({
                'type':'box',
                'size': size,
                'pos': pos,
                'vel': vel,
                'orientation': q,
                'omega': omega,
                'mass': mass,
                'inv_mass': 1.0/mass,
                'I_body': I_body,
                'I_body_inv': I_body_inv,
                'restitution': RESTITUTION_DEFAULT,
                # cached per-step:
                'R': np.eye(3),
                'half': 0.5*size,
                'corners': np.zeros((8,3), dtype=float),
                'aabb_min': np.zeros(3, dtype=float),
                'aabb_max': np.zeros(3, dtype=float),
                'I_inv_world': np.eye(3)
            })
        else:
            r = random_sphere_radius()
            vol = (4.0/3.0) * math.pi * r**3
            mass = max(0.0001, DENSITY * vol)
            pos = np.array([
                random.uniform(BOUNDS_MIN[0]+6.0, BOUNDS_MAX[0]-6.0),
                random.uniform(BOUNDS_MIN[1]+6.0, BOUNDS_MAX[1]-6.0),
                random.uniform(BOUNDS_MIN[2]+6.0, BOUNDS_MAX[2]-6.0)
            ], dtype=float)
            vel = np.array([random.uniform(-6,6), random.uniform(-6,6), random.uniform(-6,6)], dtype=float)
            axis = np.random.normal(size=3)
            n = np.linalg.norm(axis)
            axis = axis / (n + 1e-12)
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
                'restitution': RESTITUTION_DEFAULT,
                # cached per-step:
                'R': np.eye(3),
                'half': np.array([r, r, r], dtype=float),
                'corners': None,  # not used for sphere
                'aabb_min': np.zeros(3, dtype=float),
                'aabb_max': np.zeros(3, dtype=float),
                'I_inv_world': np.eye(3)
            })

# ----------------------
# Geometry helpers
# ----------------------
BOX_LOCAL_CORNERS = np.array([
    [ 1,  1,  1],
    [ 1, -1,  1],
    [-1, -1,  1],
    [-1,  1,  1],
    [ 1,  1, -1],
    [ 1, -1, -1],
    [-1, -1, -1],
    [-1,  1, -1],
], dtype=float)

def compute_box_corners_world(obj):
    # obj['half'], obj['R'], obj['pos'] must be set
    scaled = BOX_LOCAL_CORNERS * obj['half']  # (8,3)
    obj['corners'] = (obj['R'] @ scaled.T).T + obj['pos']  # (8,3)

def update_aabb_from_corners(obj):
    c = obj['corners']
    obj['aabb_min'][:] = c.min(axis=0)
    obj['aabb_max'][:] = c.max(axis=0)

def update_sphere_aabb(obj):
    r = obj['radius']
    obj['aabb_min'][:] = obj['pos'] - r
    obj['aabb_max'][:] = obj['pos'] + r

def project_points_on_axis(points, axis):
    # axis normalized before call
    projs = points @ axis
    return np.min(projs), np.max(projs)

# ----------------------
# Broadphase
# ----------------------
def aabb_overlap(a_min, a_max, b_min, b_max):
    if a_max[0] < b_min[0] or a_min[0] > b_max[0]:
        return False
    if a_max[1] < b_min[1] or a_min[1] > b_max[1]:
        return False
    if a_max[2] < b_min[2] or a_min[2] > b_max[2]:
        return False
    return True

def build_broadphase_pairs(objs):
    pairs = []
    n = len(objs)
    for i in range(n):
        Ai = objs[i]
        for j in range(i+1, n):
            Bj = objs[j]
            if aabb_overlap(Ai['aabb_min'], Ai['aabb_max'], Bj['aabb_min'], Bj['aabb_max']):
                pairs.append((i, j))
    return pairs

# ----------------------
# Collision detection
# ----------------------
def sphere_sphere_collision(a, b):
    diff = b['pos'] - a['pos']
    dist2 = float(diff.dot(diff))
    rsum = a['radius'] + b['radius']
    rsum2 = rsum * rsum
    if dist2 >= rsum2:
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
    # Compute closest point on OBB to sphere center
    R = b['R']  # cached
    local_center = R.T @ (s['pos'] - b['pos'])
    half = b['half']
    closest_local = np.minimum(np.maximum(local_center, -half), half)
    closest_world = b['pos'] + (R @ closest_local)
    diff = s['pos'] - closest_world
    dist2 = float(diff.dot(diff))
    r = s['radius']
    if dist2 >= (r*r):
        return False, None, 0.0, None
    dist = math.sqrt(dist2) if dist2 > 1e-12 else 0.0
    if dist > 1e-12:
        normal = diff / dist
    else:
        # center inside OBB: choose nearest face normal in local
        dists = np.array([half[0] - abs(local_center[0]),
                          half[1] - abs(local_center[1]),
                          half[2] - abs(local_center[2])], dtype=float)
        idx = int(np.argmin(dists))
        n_local = np.zeros(3, dtype=float)
        n_local[idx] = 1.0 if local_center[idx] >= 0 else -1.0
        normal = R @ n_local
        nrm = np.linalg.norm(normal)
        if nrm > 1e-12:
            normal /= nrm
        else:
            normal[:] = np.array([1.0,0.0,0.0], dtype=float)
    penetration = r - dist
    contact_point = closest_world
    return True, normal, penetration, contact_point

def obb_obb_collision(a, b):
    # SAT using face normals and edge cross products
    a_corners = a['corners']
    b_corners = b['corners']
    A = a['R']; B = b['R']

    axes = []
    # face normals
    axes.append(A[:,0].copy()); axes.append(A[:,1].copy()); axes.append(A[:,2].copy())
    axes.append(B[:,0].copy()); axes.append(B[:,1].copy()); axes.append(B[:,2].copy())
    # edge cross products
    for i in range(3):
        ai = A[:,i]
        for j in range(3):
            bj = B[:,j]
            cp = np.cross(ai, bj)
            nrm = np.linalg.norm(cp)
            if nrm > 1e-4:  # reject near-parallel edges
                axes.append(cp / nrm)

    min_pen = float('inf')
    min_axis = None

    center_dir = b['pos'] - a['pos']
    for axis in axes:
        nrm = np.linalg.norm(axis)
        if nrm < 1e-10:
            continue
        axis_n = axis / nrm
        a_min, a_max = project_points_on_axis(a_corners, axis_n)
        b_min, b_max = project_points_on_axis(b_corners, axis_n)
        overlap = min(a_max, b_max) - max(a_min, b_min)
        if overlap <= 0.0:
            return False, None, 0.0, None
        if overlap < min_pen:
            min_pen = overlap
            # ensure axis points from a to b
            min_axis = axis_n if center_dir.dot(axis_n) >= 0.0 else -axis_n

    # Approximate contact at midpoint of centers pushed along normal
    contact_point = 0.5*(a['pos'] + b['pos']) - min_axis * (min_pen * 0.5)
    return True, min_axis, float(min_pen), contact_point

# ----------------------
# Collision resolution
# ----------------------
def apply_impulse(a, b, contact_point, normal, penetration):
    ra = contact_point - a['pos']
    rb = contact_point - b['pos']

    va = a['vel'] + np.cross(a['omega'], ra)
    vb = b['vel'] + np.cross(b['omega'], rb)
    rel = vb - va
    vel_along_normal = float(rel.dot(normal))
    if vel_along_normal > 0.0:
        return

    Ia_inv = a['I_inv_world']
    Ib_inv = b['I_inv_world']

    inv_mass_sum = a['inv_mass'] + b['inv_mass']
    ra_x_n = np.cross(ra, normal)
    rb_x_n = np.cross(rb, normal)
    rot_term_vec = np.cross(Ia_inv @ ra_x_n, ra) + np.cross(Ib_inv @ rb_x_n, rb)
    rot_term = float(normal.dot(rot_term_vec))
    denom = inv_mass_sum + rot_term
    if denom <= 1e-12:
        return

    e = min(a.get('restitution', RESTITUTION_DEFAULT), b.get('restitution', RESTITUTION_DEFAULT))
    j = -(1.0 + e) * vel_along_normal / denom
    impulse = j * normal

    a['vel'] -= impulse * a['inv_mass']
    b['vel'] += impulse * b['inv_mass']

    a['omega'] -= Ia_inv @ np.cross(ra, impulse)
    b['omega'] += Ib_inv @ np.cross(rb, impulse)

    # positional correction
    corr_mag = max(penetration - PENETRATION_SLOP, 0.0) / (a['inv_mass'] + b['inv_mass'])
    correction = (corr_mag * POSITION_CORRECTION) * normal
    a['pos'] -= correction * a['inv_mass']
    b['pos'] += correction * b['inv_mass']

# ----------------------
# Physics integrator
# ----------------------
def cache_per_object(obj):
    # rotation, world inertia inverse, half, corners, aabb
    q = obj['orientation']
    R = quat_to_matrix(q)
    obj['R'][:] = R
    obj['I_inv_world'][:] = R @ obj['I_body_inv'] @ R.T

    if obj['type'] == 'box':
        obj['half'][:] = 0.5 * obj['size']
        compute_box_corners_world(obj)
        update_aabb_from_corners(obj)
    else:
        # sphere
        r = obj['radius']
        obj['half'][:] = r
        update_sphere_aabb(obj)

def physics_step(dt):
    # integrate
    for obj in objects:
        if obj['inv_mass'] == 0.0:
            continue
        obj['vel'] += GRAVITY * dt
        obj['pos'] += obj['vel'] * dt

        omega = obj['omega']
        q = obj['orientation']
        q_omega = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
        q_dot = 0.5 * quat_mul(q_omega, q)
        q += q_dot * dt
        quat_normalize_inplace(q)
        obj['orientation'] = q

        obj['omega'] *= 0.999  # mild damping

    # cache derived per-object data for this substep
    for obj in objects:
        cache_per_object(obj)

    # broadphase
    pairs = build_broadphase_pairs(objects)

    # narrowphase and resolution
    for (i, j) in pairs:
        A = objects[i]
        B = objects[j]
        colliding = False
        normal = None
        penetration = 0.0
        contact_point = None

        if A['type'] == 'sphere' and B['type'] == 'sphere':
            colliding, normal, penetration, contact_point = sphere_sphere_collision(A, B)
        elif A['type'] == 'sphere' and B['type'] == 'box':
            colliding, normal, penetration, contact_point = sphere_box_collision(A, B)
        elif A['type'] == 'box' and B['type'] == 'sphere':
            colliding, normal, penetration, contact_point = sphere_box_collision(B, A)
            if colliding:
                normal = -normal
        else:
            colliding, normal, penetration, contact_point = obb_obb_collision(A, B)

        if colliding and contact_point is not None:
            apply_impulse(A, B, contact_point, normal, penetration)

    # walls collisions (AABB vs bounds)
    for obj in objects:
        if obj['type'] == 'sphere':
            half = np.array([obj['radius'], obj['radius'], obj['radius']], dtype=float)
        else:
            half = obj['half']
        min_corner = obj['pos'] - half
        max_corner = obj['pos'] + half

        # X
        if min_corner[0] < BOUNDS_MIN[0]:
            obj['pos'][0] = BOUNDS_MIN[0] + half[0]
            obj['vel'][0] = -obj['vel'][0] * obj.get('restitution', RESTITUTION_DEFAULT)
            obj['omega'] *= 0.9
        elif max_corner[0] > BOUNDS_MAX[0]:
            obj['pos'][0] = BOUNDS_MAX[0] - half[0]
            obj['vel'][0] = -obj['vel'][0] * obj.get('restitution', RESTITUTION_DEFAULT)
            obj['omega'] *= 0.9
        # Y
        if min_corner[1] < BOUNDS_MIN[1]:
            obj['pos'][1] = BOUNDS_MIN[1] + half[1]
            obj['vel'][1] = -obj['vel'][1] * obj.get('restitution', RESTITUTION_DEFAULT)
            obj['omega'] *= 0.9
        elif max_corner[1] > BOUNDS_MAX[1]:
            obj['pos'][1] = BOUNDS_MAX[1] - half[1]
            obj['vel'][1] = -obj['vel'][1] * obj.get('restitution', RESTITUTION_DEFAULT)
            obj['omega'] *= 0.9
        # Z
        if min_corner[2] < BOUNDS_MIN[2]:
            obj['pos'][2] = BOUNDS_MIN[2] + half[2]
            obj['vel'][2] = -obj['vel'][2] * obj.get('restitution', RESTITUTION_DEFAULT)
            obj['omega'] *= 0.9
        elif max_corner[2] > BOUNDS_MAX[2]:
            obj['pos'][2] = BOUNDS_MAX[2] - half[2]
            obj['vel'][2] = -obj['vel'][2] * obj.get('restitution', RESTITUTION_DEFAULT)
            obj['omega'] *= 0.9

# ----------------------
# Rendering
# ----------------------
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

def draw_grid(size=140, step=GRID_STEP):
    glDisable(GL_LIGHTING)
    glColor3f(0.35, 0.35, 0.35)
    glBegin(GL_LINES)
    half = int(size * 0.5)
    for i in range(-half, half+1, step):
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

def gl_push_transform(pos, quat):
    R = quat_to_matrix(quat)
    mat = np.eye(4, dtype=float)
    mat[:3,:3] = R
    mat[:3,3] = pos
    glPushMatrix()
    glMultMatrixf(mat.T)  # column-major

# ----------------------
# OpenGL/pygame
# ----------------------
def init_opengl():
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glClearColor(0.12, 0.12, 0.12, 1.0)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)

def resize(w, h):
    if h == 0: h = 1
    glViewport(0, 0, w, h)
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
    pygame.display.set_caption("CPU Rigid Bodies: Rotating Boxes & Spheres (Optimized)")
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
                elif event.key == pygame.K_r:
                    spawn_objects()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            cam_yaw -= 50.0 * frame_dt
        if keys[pygame.K_RIGHT]:
            cam_yaw += 50.0 * frame_dt
        if keys[pygame.K_UP]:
            cam_pitch = min(89.0, cam_pitch + 50.0 * frame_dt)
        if keys[pygame.K_DOWN]:
            cam_pitch = max(-20.0, cam_pitch - 50.0 * frame_dt)
        if keys[pygame.K_w]:
            cam_distance = max(20.0, cam_distance - 60.0 * frame_dt)
        if keys[pygame.K_s]:
            cam_distance = min(150.0, cam_distance + 60.0 * frame_dt)

        # fixed-step physics with cap
        t = frame_dt
        steps = 0
        while t > 0.0 and steps < MAX_SUBSTEPS_PER_FRAME:
            step = min(DT_FIXED, t)
            physics_step(step)
            t -= step
            steps += 1

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

        draw_grid(size=140, step=GRID_STEP)
        draw_bounds()

        for obj in objects:
            gl_push_transform(obj['pos'], obj['orientation'])
            if obj['type'] == 'box':
                draw_box(obj['size'])
            else:
                draw_sphere(obj['radius'])
            glPopMatrix()

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()