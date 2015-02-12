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

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def slave(comm):
    #print("working on rank {}.".format(comm.Get_rank()))
    xlins = np.linspace(minX, maxX, width)
    lineimage = np.zeros(width, dtype=np.uint16)
    status = MPI.Status()   # get MPI status object
    while True:
        comm.send([0,0], dest=0, tag=tags.READY)
        i, y = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()

        if tag == tags.START:
	    for j,x in enumerate(xlins):
	    	lineimage[j] = mandelbrot(x,y)
            comm.send([i, lineimage], dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send([0,0], dest=0, tag=tags.EXIT)
    print "Slave finishing", comm.Get_rank()

def master(comm):
    image = np.zeros([height,width], dtype=np.uint16)
    status = MPI.Status()   # get MPI status object
    print "Master start"

    # master flow control
    ylins = np.linspace(minY, maxY, height)
    i = 0 # tasks goint to send to other processor
    iend = 0 # tasks are done
    procs = comm.Get_size() # make sure all slaves exited
 
    while iend < len(ylins) or procs > 1:
        #print "Line %d with y = %f" % (i, y)
        recv_i, recv_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if i < len(ylins) :
                comm.send([i,ylins[i]], dest=source, tag=tags.START)
		#print "sent", i,ylins[i]
                i += 1
            else:
                comm.send([0,0], dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            image[recv_i] = recv_data
            iend += 1
        elif tag == tags.EXIT: # wait for all slaves exited
	    procs -= 1
    
    print("Master finishing")
    return image

# Global variables, can be used by any process
minX,  maxX   = -2.1, 0.7
minY,  maxY   = -1.25, 1.25
width, height = 2**10, 2**10

# Define MPI message tags
tags = enum('READY', 'DONE', 'EXIT', 'START')

if __name__ == '__main__':

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    rank = comm.Get_rank()  # rank of this process
    
    start_time = time.time()
    if( rank==0 ):
        C = master(comm)
    else:
        slave(comm)

    end_time = time.time()

    if( rank==0 ):
        print "P=%d Time: %f secs" % (comm.Get_size(), end_time - start_time)
        plt.imsave('P4ms_Mandelbrot.png', C, cmap='spectral')
        plt.imshow(C, aspect='equal', cmap='spectral')
        plt.show()


