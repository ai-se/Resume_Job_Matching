# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# proc_num = 4
# result_job=[]
# if rank==0:
#     for i in range(proc_num - 1):
#         x = comm.recv(source=i + 1)
#         result_job.append(x)
#     print result_job
# else:
#     comm.send(rank, dest=0)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print rank
#
# if rank == 0:
#     data = {'a': 7, 'b': 3.14}
#     comm.send(data, dest=1, tag=11)
#     print data
# elif rank == 1:
#     data = comm.recv(source=0, tag=11)
#     print data


