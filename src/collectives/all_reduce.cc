/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, int block_num, int thread_num);

int judge(int thread_num){
  if ((thread_num % 2) == 1){
    return false;
  }
  return true;
} 

ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream, int block_num, int thread_num) {
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    // lyz -alloc
    info.block_num = block_num;//comm->block_num;
    info.thread_num = thread_num;//comm->thread_num;
    info.allreduce = 1;
    // fzh-alloc
    if (judge(thread_num)){
      info.empty = 0;
    } 
    else {
      info.empty = 1;
      info.thread_num = thread_num+1;
    }
  return ncclEnqueueCheck(&info);
}
