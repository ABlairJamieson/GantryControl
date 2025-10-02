#!/usr/bin/env python
# coding: utf-8

# # Gantry sphrical scan
# 
# Author: Blair Jamieson
# Date: December 2023
# 
# ## Gantry coordinates.
# 
# The long axis of the gantry is $x$, the other horizontal axis is $y$, and the vertical axis is $z'$.  Note that the gantry motion in $x$ and $y$ are correct, while the motion in $z$ starts at $z=z'=0$ and as the gantry $z'$ coordinate is increased $z=-z'$.
# 
# The end of the gantry's $xyz$ motion is at $\vec{r}_g$ and the location of the calibration target $\vec{r}_t'$ relative to that position depends on the rotation in the $xy$-plane $\phi_g$, and the rotation relative to the $z$ axis $\theta_g$.
# 
# $$ \vec{r}_t' = 20\ \rm{cm}\ ( \sin{\phi_g} \hat{i} + \cos{\phi_g} \hat{j} ) $$
# 
# The location of the target in gantry coordinates is:
# 
# $$ \vec{r}_t = \vec{r}_g + \vec{r}_t' $$
# 
# $$ \vec{r}_t = (x_g + 20\ \rm{cm}\ \sin{\phi_g})\hat{i} + (y_g + 20\ \rm{cm}\  \cos{\phi_g})\hat{j} - z_g'\hat{k}$$
# 
# The pointing of the calibration target (normal to the checkerboard pattern) is $\hat{n}_t$.  Note that the angle $\theta_g$ is the true spherical coordinates angle from the $z$-axis, but $\phi_g$ is measured going positive going counter-clockwise from the $y$-axis.  Therefore the true spherical coordinate $\phi = 90^{\circ} - \phi_g$.  The normal to the calibration target is:
# 
# $$\hat{n}_t = \sin{\theta_g}\cos{\phi_g}\hat{i} - \sin{\theta_g}\sin{\phi_g}\hat{j} + \cos{\theta_g}\hat{k}$$
# 
# In the gantry coordinates we define the camera position $\vec{r}_c$ and its facing $\hat{n}_c$.
# 
# ## Scan semi-sphere around camera
# 
# Suppose we want to scan a semi-spherical plane around the camera.  How do we equally space the points on the surface of the sphere?  See for example https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
# 
# The basic idea is to keep the solid angle between points constant, where
# 
# $$dA = r^2 \sin{\theta} d\theta d\phi= r d(r\cos{\theta}) d\phi = rdzd\phi$$
# 
# Choose circles of latitude at constant intervals $d_{\theta}$, and on each circle space points by distance $d_{\phi}$ such that $d_{\theta}\sim d_{\phi}$.  Suppose we want $N$ points in total.  Then:
# 
# ```
# dA = 4 * pi / N # as solid angle
# d = sqrt( dA )
# Mtheta = round( pi / d )
# dtheta = pi / Mtheta
# dphi = dA / dtheta
# for m in range(Mtheta):
#     theta = pi * ( m + 0.5 ) * m / Mtheta
#     Mphi = round( 2*pi*sin(theta) / d_theta )
#     for n in range(Mphi):
#         phi = 2 * pi * n / Mphi
#         rcvec = r * [ sin( theta )*cos(phi), 
#                        sin( theat )*sin(phi),
#                        cos( theta ) ]
# ```
# 
# Now lets consider that we don't want to scan a full sphere, since we are limited to some range of coordinates in the gantry.  Instead, lets pick a number of scan points $N$ as before, and then start and end points in $\theta$ and $\phi$.  Eg. five parameters for the scan are specified: $(N,\theta_{1},\theta_{2}, \phi_{1}, \phi_{2})$, where the ranges are chosen to allow the points in the scan to be possible for the gantry to accomplish.  
# 
# The cross sectional area covered is no longer $4\pi$, but is:
# 
# $$ dA = \Delta \phi ( \cos{ \theta_{1} } - \cos{ \theta_{2}} )$$
# $$\Delta \phi = \phi_{2} - \phi_{1}$$
# 
# The updated code becomes:
# 
# ```
# dA = (phi2-phi1) * ( cos(theta1) - cos(theta2) ) / N
# d = sqrt( dA )
# Mtheta = round( (theta2 - theta1) / d )
# dtheta = (theta2 - theta1) / Mtheta
# dphi = dA / dtheta
# for m in range(Mtheta):
#     theta = theta1 + (theta2-theta1) * ( m + 0.5 ) * m / Mtheta
#     Mphi = round( (phi2-phi1)*sin(theta) / d_theta )
#     for n in range(Mphi):
#         phi = phi1 +  (phi2-phi1) * n / Mphi
#         rcvec = rc * [ sin( theta )*cos(phi), 
#                        sin( theat )*sin(phi),
#                        cos( theta ) ]
# ```
# 

# ## Rotate from camera coords to gantry coords
# 
# In the camera coordinates, define the scan point a distance $r$ from the camera as $\vec{r'}$, and in gantry coordinates the normal to the camera as $\hat{n}$.  To keep things simple, lets assume the camera coordinate $z$-axis is the same as the gantry coordinate $z$-axis, then the rotation is just in the plane. ie. $\vec{n}$.  The angle $\phi'$ in the gantry coordinates then depends on the angle $\phi'$ between the normal and the gantry x-axis:
# 
# $$ \phi' = {\rm arctan2}{( n_y, n_x )}$$
# 
# 

# In[24]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


# ## Now calculate position and rotation of gantry
# 
# Now that we have a list of camera vectors $\vec{r}$ in gantry coordinates, we want to calculate the vector $\vec{r}_g$, $\theta_g$, and $\phi_g$ (the x,y,z of the gantry and its rotation).  This position is such that:
# 
# $$\vec{r}_g+\vec{r}_t = \vec{r}_c + \vec{r}$$
# 
# $$\vec{r}_g = \vec{r}_c + \vec{r} - \vec{r}_t$$
# 
# The vector $\vec{r}$ and $\vec{r}_c$ we get from the camera.  The vector $\vec{r}_t$ depends on the normal vector of $-\vec{r}$ (the target facing $\hat{n_t}$).  We find:
# 
# $$ \hat{n_t} = -\frac{\vec{r}}{r} $$
# $$ \phi = \rm{arctan2}( n[1] , n[0] )$$
# $$ \theta = \rm{arcsin}\left( \sqrt{ 1 - n[2]^2 } \right)$$
# 
# The gantry facing is then $\phi_g=\phi$ and $\theta_g=\theta$.  The gantry target vector is:
# 
# $$ \vec{r}_t' = 25\ \rm{cm}\ ( \sin{\phi_g} \hat{i} + \cos{\phi_g} \hat{j} ) $$
# 
# 

class camera:
    def __init__(self, rc, nc):
        '''
        Define a camera, located at vector rc (a 3-vector) in gantry coordinates
        with facing nc (a 3-vector) normal vector for the camera facing.
        '''
        self.rc = rc
        self.nc = nc
        #self.phip = np.arctan2( nc[1], nc[0] )
        
    def get_scanpoints( self, N, r, phi1, phi2, theta1, theta2  ):
        '''
        Inputs: 
        N = number of scan points
        r = distance from camera to place target
        phi1, phi2 = range in phi measured from camera point of view
        theta1, theta2 = range in polar angle theta
        Angles are in ** radians **
        
        Return a list of position vectors.
        
        rcamera = a list of approximately N points in global coordinate
                  system of the vector from the camera to the target
        '''
        assert( N > 0 )
        assert( r > 0 )
        assert( phi2 > phi1 )
        assert( theta2 > theta1 )
        rvecs = []
        dA = (phi2-phi1) * ( np.cos(theta1) - np.cos(theta2) ) / N
        d = np.sqrt( dA )
        Mtheta = int( np.rint( (theta2 - theta1) / d ) )
        dtheta = (theta2 - theta1) / Mtheta
        dphi = dA / dtheta
        for m in range(Mtheta):
            theta = theta1 + (theta2-theta1) * ( m + 0.5 ) / Mtheta
            Mphi = int( np.abs(np.rint( (phi2-phi1)*np.sin(theta) / dtheta )) )
            for n in range(Mphi):
                dphi= (phi2-phi1)* (n+0.5) / Mphi
                phi= phi1 + dphi
                #handle raster in phi
                if m % 2 == 1: #go down in phi
                    phi = phi2 - dphi

                rcvec =  [ r*np.sin( theta ) * np.cos(phi), 
                              r*np.sin( theta ) * np.sin(phi),
                              r*np.cos( theta ) ]
                rvecs.append( np.array(rcvec) )
        return rvecs

#helper functions
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0: 
        return vector
    return vector / norm

def cross_product(v1, v2):
    return np.cross(v1, v2)

#works when there is no camera roll
#FPS game type of setup.
#See LookAt matrix https://learnopengl.com/Getting-started/Camera. The transformation we are doing is inverse of LookAt matrix.
#Make sure the camera Pitch doesn't exceed +-90 degrees.
#If Pitch exceede +- 90 degrees, up should be redefined to respective upwards direction
def transform_camera_to_gantry(coordinate, camera_position, camera_direction):
    up = np.array([0, 0, 1])
    
    # Compute camera axes
    camera_left = cross_product(up, camera_direction)  # camera y-axis
    camera_up = cross_product(camera_direction, camera_left)  # camera z axis
    
    # Normalize camera axes
    x_hat = normalize(camera_direction)
    y_hat = normalize(camera_left)
    z_hat = normalize(camera_up)
    

    # Rotation matrix
    R_mat = np.array([
        [x_hat[0], y_hat[0], z_hat[0]],
        [x_hat[1], y_hat[1], z_hat[1]],
        [x_hat[2], y_hat[2], z_hat[2]]
    ])
    
    
    # Translation vector
    t_vec = np.array(camera_position)
    
    # Transform the coordinate
    coordinate = np.array(coordinate)
    transformed_coordinate = R_mat.dot(coordinate) + t_vec
    
    return transformed_coordinate

deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

def get_gantry_setting(r, rc, dc, len_rt, sensor_offset=np.zeros(3), world_to_gantry_matrix=np.eye(3)):
    """
    Given desired target position r relative to camera position rc,
    return the gantry settings (gantry_pos, phi, theta) and the target location (rt, nt)
    accounting for sensor offset at end of gantry arm.

    Parameters
    ----------
    r : np.array
        Target position in world coordinates (e.g., LED position)
    rc : np.array
        Camera position in world coordinates
    dc : np.array
        Camera direction vector
    len_rt : float
        Scalar distance along the target vector
    sensor_offset : np.array
        Sensor offset in gantry local frame (x-forward, y-left, z-up)
    world_to_gantry_matrix : np.array
        Rotation matrix from world to gantry frame

    Returns
    -------
    gantry_pos : np.array
        Gantry reference position in world coordinates
    phi : float
        Gantry yaw angle (rad)
    theta : float
        Gantry pitch angle (rad)
    rt : np.array
        Vector along gantry x-axis to sensor (length len_rt)
    nt : np.array
        Normalized pointing vector from sensor to target
    """
    # Transform target position from camera frame to gantry frame
    rg = transform_camera_to_gantry(coordinate=r, camera_position=rc, camera_direction=dc)

    # Initial vector from gantry reference to target (LED)
    nt = rc - rg
    nt /= np.linalg.norm(nt)

    # Compute preliminary gantry angles in world frame
    phig = np.pi/2 + np.arctan2(-nt[1], -nt[0])
    thetag = np.arcsin(nt[2])

    # Compute vector along gantry x-axis of length len_rt
    rthat = np.array([np.cos(phig), np.sin(phig), 0])
    rt = rthat * len_rt

    # Compute preliminary gantry position
    gantry_pos = rg - rt

    # --- Correct for sensor offset ---
    # Sensor world position
    sensor_world_pos = gantry_pos + world_to_gantry_matrix.T.dot(sensor_offset)

    # Recompute vector from sensor to LED
    nt = rc - sensor_world_pos
    nt /= np.linalg.norm(nt)

    # Convert to gantry frame
    nt_gantry = world_to_gantry_matrix.dot(nt)

    # Recompute gantry Euler angles
    phi = np.arctan2(nt_gantry[1], nt_gantry[0])
    theta = np.arcsin(-nt_gantry[2])

    return gantry_pos, phi, theta, rt, nt


def get_gantry_settings( cam, rvecs, len_rt ):
    '''
    Given a vector of scan points in rvecs, get a list of gantry settings.
    
    Inputs:
    cam = camera object
    rvecs = list of 3-vectors pointing from camera to target
    
    Returns:
    gsettings  = list of gantry settings (xg,yg,zg,phig,thetag) in cm and rad
    tlocs     = list of target positions (xt,yt,zt,ntx,nty,ntz) in cm and normal vector
    '''
    gsettings = []
    tlocs = []
    for rv in rvecs:
        rg, phig, thetag, rt, nt = get_gantry_setting( rv, cam.rc, cam.nc, len_rt )
        gsettings.append( [rg[0],rg[1],rg[2],phig,thetag] )
        tlocs.append( [rt[0],rt[1],rt[2],nt[0],nt[1],nt[2]] )
    return (gsettings, tlocs)
                           


deg2rad = np.pi / 180.0
rad2deg = 180.0 / np.pi

