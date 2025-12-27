from deepspeed import comm as ds_comm

def init_parameter_server():
    """Initialize the ParameterServer with a CDBParameterServer instance if not already set."""
    if not hasattr(ds_comm.ParameterServer, 'cd'):
        # Create a ParameterServer entry and attach a CDBParameterServer instance to it.
        parameter_server = ds_comm.ParameterServer({'parameter_name': 0})
        ds_comm.ParameterServer.cd = CDBParameterServer(parameter_server['parameter_name'])

def get_cdb_from_parameter_server():
    """Ensure the ParameterServer is initialized and return its CDB instance or None."""
    init_parameter_server()
    parameter_server = ds_comm.ParameterServer
    return getattr(parameter_server, 'cd', None)

def safe_all_reduce_parameter(parameter_key='parameter_name'):
    """
    Safe all_reduce implementation that ensures the ParameterServer and its CDB are initialized,
    then performs an all_reduce on the specified parameter and returns the first handle.
    """
    init_parameter_server()

    group = ds_comm.get_group()
    op = ds_comm.ReduceOp.SUM
    async_op = False

    parameter_server = ds_comm.ParameterServer
    cdb = getattr(parameter_server, 'cd', None)

    if cdb is None:
        print("Parameter server not properly initialized. Aborting.")
        return None

    tensor_list = [parameter_server[parameter_key]]
    output_tensor = ds_comm.all_reduce(tensor_list, op, group, async_op)
    handle0 = output_tensor[0][0]
    return handle0

# Usage example in the forward method of transformer.py
handle0 = None
handle1 = None
attention_output0 = ...  # Compute attention_output0
attention_output1 = ...  # Compute attention_output1

init_parameter_server()
handle0 = ds_comm.all_reduce([ds_comm.ParameterServer('parameter_name')], op=ds_comm.ReduceOp.SUM, group=ds_comm.get_group(), async_op=False)[0][0]
handle1 = ds_comm.all_reduce([ds_comm.ParameterServer('parameter_name')], op=ds_comm.ReduceOp.SUM, group=ds_comm.get_group(), async_op=False)[0][1]

# Now safely use handle0 and handle1 with cdb