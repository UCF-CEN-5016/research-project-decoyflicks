# Import necessary modules
from deepspeed import comm as ds_comm

def initialize_parameter_server():
    """Initialize the ParameterServer with CDB."""
    if not hasattr(ds_comm.ParameterServer, 'cd'):
        # Check if a CDB instance already exists; if not, create one.
        parameter_server = ds_comm.ParameterServer({'parameter_name': 0})
        ds_comm.ParameterServer.cd = CDBParameterServer(parameter_server['parameter_name'])
        
def all_reduce tensors across processes safely in distributed mode:
    """Safe all_reduce implementation ensuring cdb is initialized."""
    # Ensure ParameterServer and CDB are initialized
    initialize_parameter_server()
    
    group = ds_comm.get_group()
    op = ds_comm.ReduceOp.SUM
    async_op = False
    
    # Retrieve the CDB instance if it exists after initialization
    parameter_server = ds_comm.ParameterServer
    if hasattr(parameter_server, 'cd'):
        cdb = parameter_server.cd
    else:
        cdb = None  # This should not happen if initialized properly
    
    if cdb is not None:
        # Perform all_reduce using the CDB instance
        tensor_list = [parameter_server['parameter_name']]
        output_tensor = ds_comm.all_reduce(tensor_list, op, group, async_op)
        
        # Update the parameter with the reduced value
        handle0 = output_tensor[0][0]
    else:
        print("Parameter server not properly initialized. Aborting.")
        return

# Usage example in the forward method of transformer.py
handle0 = None
handle1 = None
attention_output0 = ...  # Compute attention_output0
attention_output1 = ...  # Compute attention_output1

initialize_parameter_server()
handle0 = ds_comm.all_reduce([ds_comm.ParameterServer('parameter_name')], op=ds_comm.ReduceOp.SUM, group=ds_comm.get_group(), async_op=False)[0][0]
handle1 = ds_comm.all_reduce([ds_comm.ParameterServer('parameter_name')], op=ds_comm.ReduceOp.SUM, group=ds_comm.get_group(), async_op=False)[0][1]

# Now safely use handle0 and handle1 with cdb