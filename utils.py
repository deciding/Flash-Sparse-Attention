import torch
from contextlib import contextmanager

@contextmanager
def cuda_timer(name, verbose=True):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    class TimeCapture:
        def __init__(self):
            self.elapsed_time = 0

    time_capture = TimeCapture()

    try:
        yield time_capture
    finally:
        end_event.record()
        torch.cuda.synchronize()
        time_capture.elapsed_time = start_event.elapsed_time(end_event)
        if verbose:
            print(f"[{name}] Time: {time_capture.elapsed_time:.3f} ms")