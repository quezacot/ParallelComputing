import numpy as np
import time
from mpi4py import MPI
from Plotter3DCS205 import MeshPlotter3D, MeshPlotter3DParallel
import sys

def initial_conditions(DTDX, X, Y, cart):
  '''Construct the grid points and set the initial conditions.
  X[i,j] and Y[i,j] are the 2D coordinates of u[i,j]'''
  assert X.shape == Y.shape

  um = np.zeros(X.shape)     # u^{n-1}  "u minus"
  u  = np.zeros(X.shape)     # u^{n}    "u"
  up = np.zeros(X.shape)     # u^{n+1}  "u plus"
  # Define Ix and Iy so that 1:Ix and 1:Iy define the interior points
  Ix = u.shape[0] - 1
  Iy = u.shape[1] - 1
  # Set the interior points: Initial condition is Gaussian
  u[1:Ix,1:Iy] = np.exp(-50 * (X[1:Ix,1:Iy]**2 + Y[1:Ix,1:Iy]**2))
  # Set the ghost points to the boundary conditions
  set_ghost_points(u, cart)
  # Set the initial time derivative to zero by running backwards
  apply_stencil(DTDX, um, u, up)
  set_ghost_points(u, cart)
  um *= 0.5
  # Done initializing up, u, and um
  return up, u, um

def apply_stencil(DTDX, up, u, um):
  '''Apply the computational stencil to compute u^{n+1} -- "up".
  Assumes the ghost points exist and are set to the correct values.'''

  # Define Ix and Iy so that 1:Ix and 1:Iy define the interior points
  Ix = u.shape[0] - 1
  Iy = u.shape[1] - 1
  # Update interior grid points with vectorized stencil
  up[1:Ix,1:Iy] = ((2-4*DTDX)*u[1:Ix,1:Iy] - um[1:Ix,1:Iy]
                   + DTDX*(u[0:Ix-1,1:Iy  ] +
                           u[2:Ix+1,1:Iy  ] +
                           u[1:Ix  ,0:Iy-1] +
                           u[1:Ix  ,2:Iy+1]))

  # The above is a vectorized operation for the simple for-loops:
  #for i in range(1,Ix):
  #  for j in range(1,Iy):
  #    up[i,j] = ((2-4*DTDX)*u[i,j] - um[i,j]
  #               + DTDX*(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]))


def set_ghost_points(u, cart):
    '''Set the ghost points.
    In parallel, each process will have ghost points:
      some will need data from neighboring processes,
      others will use these boundary conditions.'''
    rank = cart.Get_rank()
    coord = cart.Get_coords(rank)
    nrow, ncol = coord
    
    up, down = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    #up
    u[-1,:] = cart.sendrecv(sendobj=u[1,:], dest=up, source=down)
    #down
    u[0,:] = cart.sendrecv(sendobj=u[-2,:], dest=down, source=up)
    #left
    u[:,-1] = cart.sendrecv(sendobj=u[:,1], dest=left, source=right)
    #right
    u[:,0] = cart.sendrecv(sendobj=u[:,-2], dest=right, source=left)
    
    # The real boundaries
    # Define nx and ny so that nx+1 and ny+1 are the ghost points
    nx = u.shape[0] - 2
    ny = u.shape[1] - 2
    # Update ghost points with boundary condition
    if nrow == 0:
        u[0,:]    = u[2,:];       # u_{0,j}    = u_{2,j}      x = 0
    if nrow == Px-1:
        u[nx+1,:] = u[nx-1,:];    # u_{nx+1,j} = u_{nx-1,j}   x = 1
    if ncol == 0:
        u[:,0]    = u[:,2];       # u_{i,0}    = u_{i,2}      y = 0
    if ncol == Py-1:
        u[:,ny+1] = u[:,ny-1];    # u_{i,ny+1} = u_{i,ny-1}   y = 1


if __name__ == '__main__':
    # Global constants
    # input Px and Py
    Px = int(sys.argv[1])
    Py = int(sys.argv[2])
    xMin, xMax = 0.0, 1.0     # Domain boundaries
    yMin, yMax = 0.0, 1.0     # Domain boundaries
    Nx = 1280                 # Number of total grid points in x
    Ny = 640                  # Number of total grid points in y
    dx = (xMax-xMin)/(Nx-1)   # Grid spacing, Delta x
    dy = (yMax-yMin)/(Ny-1)   # Grid spacing, Delta y
    dt = 0.4 * dx             # Time step (Magic factor of 0.4)
    T = 5                     # Time end
    DTDX = (dt*dt) / (dx*dx)  # Precomputed CFL scalar
    Nxpart = Nx/Px
    Nypart = Ny/Py
   
    comm = MPI.COMM_WORLD
    cart = comm.Create_cart([Px, Py])
    rank = cart.Get_rank()
    coord = cart.Get_coords(rank)
    size = comm.Get_size()
    nrow, ncol = coord
    
    comm.barrier()
    p_start = MPI.Wtime()   
 
    assert Px*Py <= size
    # The global indices: I[i,j] and J[i,j] are indices of u[i,j]
    [I,J] = np.mgrid[ nrow*Nxpart-1 : (nrow+1)*Nxpart+1, ncol*Nypart-1 : (ncol+1)*Nypart+1]

    # Convenience so u[1:Ix,1:Iy] are all interior points
    Ix, Iy = Nxpart+1, Nypart+1

    # Set the initial conditions
    up, u, um = initial_conditions(DTDX, I*dx-0.5, J*dy, cart)
    print u.shape
    # Setup the serial plotter -- one plot per process
    # plotter = MeshPlotter3D()
    # Setup the parallel plotter -- one plot gathered from all processes
    plotter = MeshPlotter3DParallel()

    for k,t in enumerate(np.arange(0,T,dt)):
        # Compute u^{n+1} with the computational stencil
        apply_stencil(DTDX, up, u, um)

        # Set the ghost points on u^{n+1}
        set_ghost_points(up, cart)

        # Swap references for the next step
        # u^{n-1} <- u^{n}
        # u^{n}   <- u^{n+1}
        # u^{n+1} <- u^{n-1} to be overwritten in next step
        um, u, up = u, up, um

        # Output and draw Occasionally

        if k % 100 == 0:
            print "Step: %d  Time: %f" % (k,t)
            #plotter.draw_now(I[1:Ix,1:Iy], J[1:Ix,1:Iy], u[1:Ix,1:Iy])

    comm.barrier()
    p_stop = MPI.Wtime()
    if rank==0:
        print "P=", size, "(Nx,Ny)=", Nx,Ny, "(Px,Py)=", Px,Py
        print "Parallel Time: %f secs" % (p_stop - p_start)
    plotter.save_now(I, J, u, "P3a-FinalWave"+ str(Px) + "_" + str(Py) +".png")


