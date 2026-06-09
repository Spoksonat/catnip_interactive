import numpy as np
import sympy as sp

def R_x(theta):
    """
    Returns the rotation matrix in the special Euclidean group SE(3) for a rotation around the X axis.

    Args:
        theta (float): Rotation angle in degrees.

    Returns:
        sympy.Matrix: Rotation matrix in SE(3).
    """
    theta = np.deg2rad(theta)
    return sp.Matrix([
        [1, 0, 0, 0],
        [0, sp.cos(theta), -sp.sin(theta), 0],
        [0, sp.sin(theta),  sp.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def R_y(theta):
    """
    Returns the rotation matrix in the special Euclidean group SE(3) for a rotation around the Y axis.

    Args:
        theta (float): Rotation angle in degrees.

    Returns:
        sympy.Matrix: Rotation matrix in SE(3).
    """
    theta = np.deg2rad(theta)
    return sp.Matrix([
        [sp.cos(theta), 0, sp.sin(theta), 0],
        [0, 1, 0, 0],
        [-sp.sin(theta), 0, sp.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def R_z(theta):
    """
    Returns the rotation matrix in the special Euclidean group SE(3) for a rotation around the Z axis.

    Args:
        theta (float): Rotation angle in degrees.

    Returns:
        sympy.Matrix: Rotation matrix in SE(3).
    """
    theta = np.deg2rad(theta)
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta), 0, 0],
        [sp.sin(theta),  sp.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def T(x_0, y_0, z_0):
    """
    Returns the translation matrix in the special Euclidean group SE(3) for a translation by (x_0, y_0, z_0).

    Args:
        x_0 (float): Translation in X.
        y_0 (float): Translation in Y.
        z_0 (float): Translation in Z.

    Returns:
        sympy.Matrix: Translation matrix in SE(3).
    """
    return sp.Matrix([
        [1, 0, 0, -x_0],
        [0, 1, 0, -y_0],
        [0, 0, 1,  -z_0],
        [0, 0, 0, 1]
    ])

def R_x_np(theta):
    """
    Returns the rotation matrix in the special Euclidean group SE(3) for a rotation around the X axis.

    Args:
        theta (float): Rotation angle in degrees.

    Returns:
        Matrix: Rotation matrix in SE(3).
    """
    theta = np.deg2rad(theta)
    return np.array([[1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0, 0, 1]
        ])

def R_y_np(theta):
    """
    Returns the rotation matrix in the special Euclidean group SE(3) for a rotation around the Y axis.

    Args:
        theta (float): Rotation angle in degrees.

    Returns:
        Matrix: Rotation matrix in SE(3).
    """
    theta = np.deg2rad(theta)
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])

def R_z_np(theta):
    """
    Returns the rotation matrix in the special Euclidean group SE(3) for a rotation around the Z axis.

    Args:
        theta (float): Rotation angle in degrees.

    Returns:
        Matrix: Rotation matrix in SE(3).
    """
    theta = np.deg2rad(theta)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def T_np(x_0, y_0, z_0):
    """
    Returns the translation matrix in the special Euclidean group SE(3) for a translation by (x_0, y_0, z_0).

    Args:
        x_0 (float): Translation in X.
        y_0 (float): Translation in Y.
        z_0 (float): Translation in Z.

    Returns:
        Matrix: Translation matrix in SE(3).
    """
    return np.array([
        [1, 0, 0, -x_0],
        [0, 1, 0, -y_0],
        [0, 0, 1,  -z_0],
        [0, 0, 0, 1]
    ])

def preamble(img_size, f, R_microst, Volume_tot, cx, cy, cz):
    Volume_microst = (4/3)*np.pi*(R_microst**3) 
    print("R micro: ", R_microst)
    N_spheres = int(f*Volume_tot/Volume_microst)
    Nx, Ny = img_size
    print(N_spheres)

    if(cx==None):
        cx = img_size[1]//2

    if(cy==None):
        cy = img_size[0]//2

    if(cz==None):
        cz = 0

    # Grid
    x = np.linspace(0, Nx, Nx)
    y = np.linspace(0, Ny, Ny)
    XX, YY = np.meshgrid(x, y)

    thickness = np.zeros_like(XX)

    return N_spheres, XX, YY, thickness, cx, cy, cz

def rotation(theta_y, theta_z, img_size, cx, cy, cz, cx_m, cy_m, cz_m, XX, YY, thickness, R_microst):
    Rz_proper = R_z_np(theta_z)
    Ry_fixed_translated = np.dot(T_np(x_0 = -(img_size[0]/2 - cx), y_0=0, z_0=0),  np.dot(R_y_np(theta_y), T_np(x_0 = (img_size[0]/2 - cx), y_0=0, z_0=0) ) ) 
    Ry_proper = np.dot(Rz_proper, np.dot(Ry_fixed_translated, Rz_proper.T))
    R_total = np.dot(Ry_proper, Rz_proper)

    vs = np.array([[x,y,z,1] for x,y,z in zip(cx_m,cy_m,cz_m)])

    vs_rot = np.zeros_like(vs)

    for i, v in enumerate(vs): 
        v_t = np.dot(R_total,v)
        vs_rot[i] = np.array([v_t[0] + cx, v_t[1] + cy, v_t[2] + cz, v_t[3]])

    cx_m_rot, cy_m_rot = vs_rot[:,0], vs_rot[:,1]

    # Superposicion de thickness
    for x0, y0 in zip(cx_m_rot, cy_m_rot):
        r2 = (XX - x0)**2 + (YY - y0)**2
        mask = r2 <= R_microst**2
        thickness[mask] += 2.0 * np.sqrt(R_microst**2 - r2[mask])

    return thickness

def geometric_variables(theta_y, theta_z, img_size, cx=None, cy=None, cz=None):
    """
    Computes geometric variables and rotated coordinates for a given image size and rotation.

    Args:
        theta_y (float): Rotation angle around Y axis in degrees.
        theta_z (float): Rotation angle around Z axis in degrees.
        img_size (tuple): Image size (height, width).
        cx, cy, cz (int, optional): Center coordinates. Defaults to image center.

    Returns:
        tuple: Geometric variables including meshgrid, symbols, and rotated coordinates.
    """
    if(cx==None):
        cx = img_size[1]//2

    if(cy==None):
        cy = img_size[0]//2

    if(cz==None):
        cz = 0 

    Y, X = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]), indexing='ij')
    
    x, y, z = sp.symbols('x y z', real=True)
    
    v = sp.Matrix([x-cx, y-cy, z-cz, 1])
    Rz_proper = R_z(theta_z)
    Ry_fixed_translated = T(x_0 = -(img_size[0]/2 - cx), y_0=0, z_0=0) * R_y(theta_y) * T(x_0 = (img_size[0]/2 - cx), y_0=0, z_0=0)
    Ry_proper = Rz_proper * Ry_fixed_translated * Rz_proper.T
    R_total = Ry_proper * Rz_proper
    v_rot =  R_total * v
    x_r, y_r, z_r, _ = v_rot

    return cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r

def sympy_to_numpy(projection, img_size, x, y, X, Y):
    """
    Converts a sympy expression to a numpy array for a given image size.

    Args:
        projection (sympy expression): Expression to convert.
        img_size (tuple): Image size.
        x, y (sympy.Symbol): Symbolic variables.
        X, Y (np.ndarray): Meshgrid arrays.

    Returns:
        np.ndarray: Evaluated projection as a numpy array.
    """
    # Simplify
    projection = sp.simplify(projection)
    
    P_func = sp.lambdify((x, y), projection, modules='numpy')
    
    # --- Initialize image ---
    projection = np.zeros(img_size, dtype=np.float32)
    
    # --- Evaluate only inside the valid region ---
    with np.errstate(divide='ignore', invalid='ignore'):
        projection = P_func(X, Y)
    projection = np.nan_to_num(projection)

    return projection


def create_ellipse_proj(theta_y, theta_z, img_size, a, b, c, cx=None, cy=None, cz=None):
    """
    Creates a projection of an ellipse (or sphere) with given parameters and rotation.

    Args:
        theta_y (float): Rotation angle around Y axis in degrees.
        theta_z (float): Rotation angle around Z axis in degrees.
        img_size (tuple): Image size.
        a, b, c (float): Semi-axes lengths.
        cx, cy, cz (int, optional): Center coordinates.

    Returns:
        np.ndarray: 2D projection of the ellipse.
    """
    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Sphere equation inside indicator function
    sphere_expr = (x_r/a)**2 + (y_r/b)**2 + (z_r/c)**2 - 1.0
    
    z_limits = sp.solve(sphere_expr, z) # z values for where sphere_expr = 0
    z_lower, z_upper = z_limits[0], z_limits[1]
    
    # Integrate 1 over z between those limits
    projection = sp.integrate(1, (z, z_lower, z_upper))
    
    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_box_proj(theta_y, theta_z, img_size, Lx, Ly, Lz, cx=None, cy=None, cz=None):
    """
    Creates a projection of a box (cuboid) with given parameters and rotation.

    Args:
        theta_y (float): Rotation angle around Y axis in degrees.
        theta_z (float): Rotation angle around Z axis in degrees.
        img_size (tuple): Image size.
        Lx, Ly, Lz (float): Box dimensions.
        cx, cy, cz (int, optional): Center coordinates.

    Returns:
        np.ndarray: 2D projection of the box.
    """
    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Cube indicator (1 inside cube, 0 outside)
    inside_cube = sp.Piecewise(
        (1,
         (x_r >= -Lx/2) & (x_r <= Lx/2) &
         (y_r >= -Ly/2) & (y_r <= Ly/2) &
         (z_r >= -Lz/2) & (z_r <= Lz/2)),
        (0, True)
    )
    
    # Integrate over z from -∞ to ∞ (outside cube indicator=0 so integral converges)
    projection = sp.integrate(inside_cube, (z, -sp.oo, sp.oo))
    
    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_wedge_proj(theta_y, theta_z, img_size, Lx, Ly, Lz, cx=None, cy=None, cz=None):
    """
    Creates a projection of a wedge with given parameters and rotation.

    Args:
        theta_y (float): Rotation angle around Y axis in degrees.
        theta_z (float): Rotation angle around Z axis in degrees.
        img_size (tuple): Image size.
        Lx, Ly, Lz (float): Wedge dimensions.
        cx, cy, cz (int, optional): Center coordinates.

    Returns:
        np.ndarray: 2D projection of the wedge.
    """
    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Sphere equation inside indicator function
    wedge_low_expr = z_r + Lz/2
    wedge_high_expr = z_r - Lz/2
    
    inside_cube = sp.Piecewise(
        (1,
        (x_r >= -Lx/2) & (x_r <= Lx/2) &
        (y_r >= -Ly/2) & (y_r <= Ly/2)),
        (0, True)
    )

    z_limits_low = sp.solve(wedge_low_expr, z) # z values for where sphere_expr = 0
    z_lower = z_limits_low[0]

    z_limits_high = sp.solve(wedge_high_expr, z) # z values for where sphere_expr = 0
    z_upper = z_limits_high[0]
    
    if(theta_y <= 90):
        projection = sp.integrate(inside_cube, (z, z_lower, -(Lz/Lx)*x_r) )
    else:
        projection = sp.integrate(inside_cube, (z, -(Lz/Lx)*x_r, z_upper) )

    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_cylinder_proj(theta_y, theta_z, img_size, D, h, cx=None, cy=None, cz=None):
    """
    Creates a projection of a cylinder with given parameters and rotation.

    Args:
        theta_y (float): Rotation angle around Y axis in degrees.
        theta_z (float): Rotation angle around Z axis in degrees.
        img_size (tuple): Image size.
        D (float): Diameter of the cylinder.
        h (float): Height of the cylinder.
        cx, cy, cz (int, optional): Center coordinates.

    Returns:
        np.ndarray: 2D projection of the cylinder.
    """
    cx, cy, cz, Y, X, x, y, z, x_r, y_r, z_r = geometric_variables(theta_y, theta_z, img_size, cx, cy, cz)
    
    # Sphere equation inside indicator function
    sphere_expr = (x_r/(D/2))**2 + (y_r/(100*D))**2 + (z_r/(D/2))**2 - 1.0

    # Cube indicator (1 inside cube, 0 outside)
    inside_cube = sp.Piecewise(
        (1,
         (x_r >= -D/2) & (x_r <= D/2) &
         (y_r >= -h/2) & (y_r <= h/2) &
         (z_r >= -D/2) & (z_r <= D/2)),
        (0, True)
    )
    
    z_limits = sp.solve(sphere_expr, z) # z values for where sphere_expr = 0
    z_lower, z_upper = z_limits[0], z_limits[1]
    
    # Integrate 1 over z between those limits
    projection = sp.integrate(inside_cube, (z, z_lower, z_upper))#*y_indicator
    
    projection = sympy_to_numpy(projection, img_size, x, y, X, Y)

    return projection

def create_hollow_cylinder_proj(theta_y, theta_z, img_size, D, D_int, h, h_int, cx=None, cy=None, cz=None):
    """
    Creates a projection of a hollow cylinder by subtracting inner cylinder from outer cylinder.

    Args:
        theta_y (float): Rotation angle around Y axis in degrees.
        theta_z (float): Rotation angle around Z axis in degrees.
        img_size (tuple): Image size.
        D (float): Outer diameter.
        D_int (float): Inner diameter.
        h (float): Outer height.
        h_int (float): Inner height.
        cx, cy, cz (int, optional): Center coordinates.

    Returns:
        np.ndarray: 2D projection of the hollow cylinder.
    """
    cylinder_projection = create_cylinder_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, D = D, h = h, cx=cx, cy=cy, cz=cz)
    cylinder_int_projection = create_cylinder_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, D = D_int, h = h_int, cx=cx, cy=cy, cz=cz)

    hollow_cylinder_projection = cylinder_projection - cylinder_int_projection

    return hollow_cylinder_projection

def create_hollow_cube_proj(theta_y, theta_z, img_size, Lx, Ly, Lz, Lx_int, Ly_int, Lz_int, cx=None, cy=None, cz=None):
    """
    Creates a projection of a hollow cube by subtracting inner box from outer box.

    Args:
        theta_y (float): Rotation angle around Y axis in degrees.
        theta_z (float): Rotation angle around Z axis in degrees.
        img_size (tuple): Image size.
        Lx, Ly, Lz (float): Outer box dimensions.
        Lx_int, Ly_int, Lz_int (float): Inner box dimensions.
        cx, cy, cz (int, optional): Center coordinates.

    Returns:
        np.ndarray: 2D projection of the hollow cube.
    """
    box_projection = create_box_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, Lx=Lx, Ly=Ly, Lz=Lz, cx=cx, cy=cy, cz=cz)
    box_int_projection = create_box_proj(theta_y=theta_y, theta_z = theta_z, img_size=img_size, Lx=Lx_int, Ly=Ly_int, Lz=Lz_int, cx=cx, cy=cy, cz=cz)

    hollow_box_projection = box_projection - box_int_projection

    return hollow_box_projection

def create_cylinder_proj_DF(theta_y, theta_z, img_size, binning_factor, D, h, f, R_microst, cx=None, cy=None, cz=None):

    #img_size = (img_size[0]//binning_factor, img_size[1]//binning_factor)
    R_cyl = D/2
    Volume_tot = np.pi*(R_cyl**2)*h
    print("Volume tot: ",Volume_tot)
    
    N_spheres, XX, YY, thickness, cx, cy, cz = preamble(img_size, f, R_microst, Volume_tot, cx, cy, cz)
    # Centros aleatorios en cilindro
    rng = np.random.default_rng(42)

    r = R_cyl * np.sqrt(rng.random(N_spheres))
    phi = 2 * np.pi * rng.random(N_spheres)

    cx_m = r * np.cos(phi)
    cz_m = r * np.sin(phi)
    cy_m = rng.uniform(-h / 2, h / 2, N_spheres)

    thickness = rotation(theta_y, theta_z, img_size, cx, cy, cz, cx_m, cy_m, cz_m, XX, YY, thickness, R_microst)
    #thickness = np.repeat(np.repeat(thickness, binning_factor, axis=0), binning_factor, axis=1)

    return thickness

def create_box_proj_DF(theta_y, theta_z, img_size, binning_factor, Lx, Ly, Lz, f, R_microst, cx=None, cy=None, cz=None):
    #img_size = (img_size[0]//binning_factor, img_size[1]//binning_factor)

    Volume_tot = Lx*Ly*Lz
    N_spheres, XX, YY, thickness, cx, cy, cz = preamble(img_size, f, R_microst, Volume_tot, cx, cy, cz)

    # Centros aleatorios en cilindro
    rng = np.random.default_rng(42)

    cx_m = Lx*rng.random(N_spheres) - Lx/2
    cz_m = Lz*rng.random(N_spheres) - Lz/2
    cy_m = Ly*rng.random(N_spheres) - Ly/2

    thickness = rotation(theta_y, theta_z, img_size, cx, cy, cz, cx_m, cy_m, cz_m, XX, YY, thickness, R_microst)
    #thickness = np.repeat(np.repeat(thickness, binning_factor, axis=0), binning_factor, axis=1)
    return thickness

def create_ellipse_proj_DF(theta_y, theta_z, img_size, binning_factor, a, b, c, f, R_microst, cx=None, cy=None, cz=None):
    #img_size = (img_size[0]//binning_factor, img_size[1]//binning_factor)

    Volume_tot = (4/3)*np.pi*a*b*c
    N_spheres, XX, YY, thickness, cx, cy, cz = preamble(img_size, f, R_microst, Volume_tot, cx, cy, cz)

    # Centros aleatorios en cilindro
    rng = np.random.default_rng(42)

    r_m = rng.random(N_spheres)**(1/3)
    u = np.random.normal(size=(N_spheres, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    a_m, b_m, c_m = a * r_m, b * r_m, c * r_m

    cx_m = a_m * u[:,0]
    cz_m = c_m * u[:,2]
    cy_m = b_m * u[:,1]

    thickness = rotation(theta_y, theta_z, img_size, cx, cy, cz, cx_m, cy_m, cz_m, XX, YY, thickness, R_microst)
    #thickness = np.repeat(np.repeat(thickness, binning_factor, axis=0), binning_factor, axis=1)
    return thickness

def create_wedge_proj_DF(theta_y, theta_z, img_size, binning_factor, Lx, Ly, Lz, f, R_microst, cx=None, cy=None, cz=None):
    #img_size = (img_size[0]//binning_factor, img_size[1]//binning_factor)

    Volume_tot = 0.5*Lx*Ly*Lz
    N_spheres, XX, YY, thickness, cx, cy, cz = preamble(img_size, f, R_microst, Volume_tot, cx, cy, cz)

    # Centros aleatorios en cilindro
    rng = np.random.default_rng(42)

    cx_m = Lx*rng.random(N_spheres)**(1/2) - Lx/2
    cy_m = Ly*rng.random(N_spheres)**(1/2) - Ly/2

    # z limits depending on x
    z_min = -Lz * cx_m  / Lx
    z_max =  Lz / 2

    # z in [z_min, z_max]
    cz_m  = z_min + (z_max - z_min) * rng.random(N_spheres)**(1/2)
    
    thickness = rotation(theta_y, theta_z, img_size, cx, cy, cz, cx_m, cy_m, cz_m, XX, YY, thickness, R_microst)
    #thickness = np.repeat(np.repeat(thickness, binning_factor, axis=0), binning_factor, axis=1)
    return thickness

def create_hollow_cylinder_proj_DF(theta_y, theta_z, img_size, binning_factor, D, D_int, h, f, R_microst, cx=None, cy=None, cz=None):
    #img_size = (img_size[0]//binning_factor, img_size[1]//binning_factor)

    R_cyl = D/2
    Volume_tot = np.pi*(R_cyl**2)*h
    
    N_spheres, XX, YY, thickness, cx, cy, cz = preamble(img_size, f, R_microst, Volume_tot, cx, cy, cz)
    # Centros aleatorios en cilindro
    rng = np.random.default_rng(42)

    r_min = D_int/2
    r_max = D/2
    r = r_min + (r_max - r_min) * rng.random(N_spheres)**(1/2)
    phi = 2 * np.pi * rng.random(N_spheres)

    cx_m = r * np.cos(phi)
    cz_m = r * np.sin(phi)
    cy_m = rng.uniform(-h / 2, h / 2, N_spheres)

    thickness = rotation(theta_y, theta_z, img_size, cx, cy, cz, cx_m, cy_m, cz_m, XX, YY, thickness, R_microst)
    #thickness = np.repeat(np.repeat(thickness, binning_factor, axis=0), binning_factor, axis=1)
    return thickness