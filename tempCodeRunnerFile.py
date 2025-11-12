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
