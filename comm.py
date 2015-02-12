from mpi4py import MPI

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  # sets up communicator object
  comm = MPI.COMM_WORLD
  # gets rank of the processors inside thie communicator
  rank = comm.Get_rank()
  # gets number of processors in the communicator
  size = comm.Get_size()
  print "Processor %d reporting for duty!" % rank


