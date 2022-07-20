
__init__  
#########################


AbstractOptimizer.create_parameter_files(self, params: List[dict]) -> None:
#############################################################################
Create hyper parameter files.

Args:
    params (List[dict]): A list of hyper parameter dictionaries.

Returns:
    None


AbstractOptimizer.create_parameter_file(self, param: dict) -> str:
#######################################################################
Create a hyper parameter file.

Args:
    param (dict): A hyper parameter dictionary.

Returns:
    str: An unique hyper parameter name.



AbstractOptimizer.generate_initial_parameter(self) -> Union[None, Dict]
##########################################################################
Generate a initial parameter.

Returns:
    Union[None, Dict]


AbstractOptimizer.generate_parameter(self, number: Optional[int] = 1) -> None:
################################################################################
Generate parameters.

Args:
    number (Optional[int]): A number of generating parameters.

Returns:
    None

Raises:
    NotImplementedError: Causes when the inherited class does not
        implement.


AbstractOptimizer.pre_process(self) -> None:
################################################################################
Pre-procedure before executing processes.

Returns:
    None


AbstractOptimizer.post_process(self) -> None:
################################################################################
Post-procedure after executed processes.

Returns:
    None


AbstractOptimizer.loop_pre_process(self) -> None:
################################################################################
Called before entering a main loop process.

Returns:
    None


AbstractOptimizer.loop_post_process(self) -> None:
################################################################################
Called after exiting a main loop process.

Returns:
    None


AbstractOptimizer.inner_loop_pre_process(self) -> bool:
################################################################################
Called before executing a main loop process.
This process is repeated every main loop.

Returns:
    bool: The process succeeds or not. The main loop exits if failed.



AbstractOptimizer.inner_loop_main_process(self) -> bool:
################################################################################
A main loop process. This process is repeated every main loop.

Returns:
    bool: The process succeeds or not. The main loop exits if failed.


AbstractOptimizer.inner_loop_post_process(self) -> bool:
################################################################################
Called after exiting a main loop process. This process is repeated every main loop.

Returns:
    bool: The process succeeds or not. The main loop exits if failed.


AbstractOptimizer._serialize(self) -> dict:
################################################################################
Serialize this module.

Returns:
    None


AbstractOptimizer._deserialize(self, dict_objects: dict) -> None:
################################################################################
Deserialize this module.

Args:
    dict_objects(dict): A dictionary including serialized objects.

Returns:
    None
