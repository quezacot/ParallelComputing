from P2serial import data_transformer, serialtran
from mpi4py import MPI
import numpy as np
import time
import matplotlib.pyplot as plt
import math
plt.ion()         # Allow interactive updates to the plots

def taskdevide(k, rank):
    an = int(k/rank)
    l = range(0, k, an)
    rest = l[-1] - k
    if( rest == -1*an ):
        l.append(k)
    else:
        for i,j in enumerate(range(rest, 0)):
            l[j] += i+1
    return l

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def paralleltran(nrow, ncol, imgsize, data, tri, comm):
  imposed = np.zeros(imgsize*imgsize).reshape((imgsize,imgsize))  
  
  # Define MPI message tags
  tags = enum('READY', 'DONE', 'EXIT', 'START')

  # Initializations and preliminaries
  size = comm.Get_size()  # total number of processes
  rank = comm.Get_rank()  # rank of this process
  status = MPI.Status()   # get MPI status object
  
  task_index = 0
  num_tasks = size - 1
  indices = taskdevide(nrow, num_tasks)
  taskind = 0
  end_tasks = 0

  if rank==0:
    #print("Master starting with {} workers".format(num_tasks))
    
    while end_tasks < num_tasks: #tasks not finished yet
        recv_data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()
        if tag == tags.READY:
            # Worker is ready, so send it a task
            if taskind+1 < len(indices):
                comm.send(taskind, dest=source, tag=tags.START)
                #print("Sending task {} to rank {}".format(taskind, source))
                taskind += 1
            else:
                comm.send(None, dest=source, tag=tags.EXIT)
        elif tag == tags.DONE:
            results = recv_data
            imposed += results
            #print("Got data from work rank {}".format(source))
        elif tag == tags.EXIT:
            #print("Worker {} exited.".format(source))
            end_tasks += 1

    #print("Master finishing")
  else:
    # Worker processes execute code below
    #print("working on rank {}.".format(rank))
    while True:
        comm.send(None, dest=0, tag=tags.READY)
        recv_ind = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        #print "receive ind", recv_ind
	tag = status.Get_tag()

        if tag == tags.START:
          result = np.zeros(imgsize*imgsize).reshape((imgsize,imgsize))
	  #print "indices", indices[recv_ind], indices[recv_ind+1]
          for k in range(indices[recv_ind], indices[recv_ind+1]):
            result += tr.transform( data[k], -1.0*k*np.pi/nrow )
	  comm.send(result, dest=0, tag=tags.DONE)
        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)

  return imposed
  
if __name__ == '__main__':

  nrow, ncol = 2048, 6144
  imgsize = 2048
  data = np.fromfile('TomoData.bin', dtype=np.dtype('d')).reshape((nrow, ncol))
  tr = data_transformer( ncol, imgsize)
  comm = MPI.COMM_WORLD   # get MPI communicator object
  rank = comm.Get_rank()  # rank of this process
  size = comm.Get_size()  # total number of processes

  comm.barrier()
  p_start = MPI.Wtime()
  p_imposed = paralleltran(nrow, ncol, imgsize, data, tr, comm)
  comm.barrier()
  p_stop = MPI.Wtime()

  if rank == 0:
    print "P=", size, "ImageSize=", imgsize
    print "Parallel Time: %f secs" % (p_stop - p_start)
    plt.imsave("img_P2a"+str(imgsize)+".jpg", p_imposed, cmap='bone')
'''
  # Check and output results on process 0
  if rank == 0:
    s_start = time.time()
    s_imposed = serialtran(nrow, ncol, imgsize, data, tr)
    s_stop = time.time()
    print "Serial Time: %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)
    rel_error = abs(p_imposed.sum() - s_imposed.sum()) / abs(s_imposed.sum())
    print "Relative Error  = %e" % rel_error
    if rel_error > 1e-10:
      print "***LARGE ERROR - POSSIBLE FAILURE!***"
'''
