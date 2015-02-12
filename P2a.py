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

def paralleltran(nrow, ncol, imgsize, data, tri, comm):
  # Initializations and preliminaries
  size = comm.Get_size()  # total number of processes
  rank = comm.Get_rank()  # rank of this process
  status = MPI.Status()   # get MPI status object
  
  indices = taskdevide(nrow, size)

  recv_id = 0
  if rank==0:
    for i in range(1, size):
      comm.send(i, dest=i)   
      #print "root sent", i
    recv_ind = 0
  else:
    recv_ind = comm.recv(source=0, tag=MPI.ANY_TAG)

  result = np.zeros(imgsize*imgsize).reshape((imgsize,imgsize))
  #print "rank", recv_id, "start"
  for k in range(indices[recv_ind], indices[recv_ind+1]):
    result += tr.transform( data[k], -1.0*k*np.pi/nrow )
  #print "rank", recv_id, "finished and send"

  imposed = np.zeros(imgsize*imgsize).reshape((imgsize,imgsize))
  if rank==0:
    imposed += result
    for i in range(1, size):
       part_result = comm.recv(source=i, tag=MPI.ANY_TAG)
       #print "root received", i
       imposed += part_result
  else:
    comm.send(result, dest=0)
  
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
