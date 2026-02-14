from transformers import TrainerCallback
import torch.distributed as dist
import torch


class ProfileCallback(TrainerCallback):
    def __init__(self, model, decay=0.9999):
        self.prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=40, warmup=2, active=2, repeat=1),
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            profile_memory=False,
        )
        self.prof.start()
    def on_step_begin(self, *_args, **_kwargs):
        self.prof.step()

    def on_step_end(self, *_args, **_kwargs):
        self.prof.step()

    def _trace_handler(self, p):
        local_file = f"rank_{dist.get_rank()}.json.gz"
        p.export_chrome_trace(local_file)
