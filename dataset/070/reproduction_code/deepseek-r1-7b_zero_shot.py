import os
from deepspeed import initialize, comm
os.environ['DEEPSpeedInitialization'] = 'init'

def init_process(args):
    ds_init = initialize()
    if not ds_init:
        return False
    print("DeepSpeed initialized with rank: {}".format(ds_init))
    args.n_replicas = int(args.num_gpus * args.num_nodes)
    
    # Initialize process group
    if args.distributed:
        comm.init_process_group(args.backend, args.node_rank,
                               args.master_addr, args.master_port,
                               world_size=args.n_replicas,
                               rank=args.process_index)

args = {
    'num_gpus': 2,
    'num_nodes': 1,
    'hostfile': '/tmp/myhostfile',
    'no_ssh': True,
    'node_rank': 0,
    'master_addr': 'localhost',
    'master_port': 8836,
    # Other deepspeed args...
}

args = initialize(args)
if not args:
    print("DeepSpeed initialization failed")
    
init_process(args)