import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI

def mandelbrot(x, y):
    '''Compute a Mandelbrot pixel -- Unoptimized'''
    z = c = complex(x,y)
    it, maxit = 0, 511
    while abs(z) < 2 and it < maxit:
        z = z*z + c
        it += 1
    return it

def parallmande(comm):
    size = comm.Get_size()
    rank = comm.Get_rank()

    datalist = []

    for i,y in enumerate(np.linspace(minY, maxY, height)):
	#print "enumerate", i, y
	if rank == i % size:
            #print "rank", rank, (i % (size/chunk)) / chunk
	    lineimage = np.zeros(width, dtype=np.uint16)
            for j,x in enumerate(np.linspace(minX, maxX, width)):
	        lineimage[j] = mandelbrot(x,y)
            datalist.append(lineimage)
    
    image = np.empty([0,width], dtype=np.uint16)
    datacoll = comm.gather(datalist, root=0)
  
    if rank == 0:
        for i in range(height/size):
    	    for j in range(size):
	        temp = np.array(datacoll[j][i])
	        image = np.vstack([image,temp])
    return image
    


# Global variables, can be used by any process
minX,  maxX   = -2.1, 0.7
minY,  maxY   = -1.25, 1.25
width, height = 2**10, 2**10

if __name__ == '__main__':

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.Get_rank()  # rank of this process
    
    start_time = time.time()
    C = parallmande(comm)
    end_time = time.time()

    if( rank==0 ):
        print "P4Sally: P=%d, Time: %f secs" % (comm.Get_size(), end_time - start_time)
        plt.imsave('P4Sally_Mandelbrot.png', C, cmap='spectral')
        plt.imshow(C, aspect='equal', cmap='spectral')
        plt.show()


