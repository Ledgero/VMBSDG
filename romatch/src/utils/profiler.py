import torch
from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler
from contextlib import contextmanager
from pytorch_lightning.utilities import rank_zero_only
import time # Added for time.time()


class InferenceProfiler(SimpleProfiler):
    """
    This profiler records duration of actions with cuda.synchronize()
    Use this in test time. 
    """

    def __init__(self, save_dir=None):
        super().__init__()
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)
        self.recorded_durations = [] # To store raw data for each profile call
        self.save_dir = save_dir # Store save_dir

    @contextmanager
    def profile(self, action_name: str) -> None:
        try:
            torch.cuda.synchronize()
            start_time = time.time() # Use time.time() for more accurate duration for each call
            # self.start(action_name) # No need to call SimpleProfiler's start/stop for raw data
            yield action_name
        finally:
            torch.cuda.synchronize()
            end_time = time.time()
            duration = end_time - start_time
            self.recorded_durations.append({"action": action_name, "duration": duration})
            # self.stop(action_name) # No need to call SimpleProfiler's start/stop for raw data

    def get_raw_data(self):
        return self.recorded_durations


def build_profiler(name, save_dir=None):
    if name == 'inference':
        return InferenceProfiler(save_dir=save_dir)
    elif name == 'pytorch':
        from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler,PyTorchProfiler
        return PyTorchProfiler(use_cuda=True, profile_memory=True, row_limit=100, dirpath=save_dir)
    elif name is None:
        return PassThroughProfiler()
    else:
        raise ValueError(f'Invalid profiler: {name}')
