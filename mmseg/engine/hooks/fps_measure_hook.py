from typing import Any, Optional, Union

import numpy as np
import torch
from mmengine import print_log
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import time
import logging

@HOOKS.register_module()
class FPSMeasureHook(Hook):
    """
    """
    priority = 'VERY_LOW'
    def __init__(self, interval=1000, n_warmup=10, total_iters = 100, repeat_times=1):
        self.interval = interval
        self.n_warmup = n_warmup
        self.repeat_times = repeat_times
        self.total_iters = total_iters


    def measure_fps(self, model, data_loader, log_interval=100000):
        benchmark_dict = {}

        overall_fps_list = []
        for time_index in range(self.repeat_times):
            print(f'Run {time_index + 1}:')
            model.eval()

            # the first several iterations may be very slow so skip them
            num_warmup = self.n_warmup
            pure_inf_time = 0
            total_iters = self.total_iters

            # benchmark with 200 batches and take the average
            for i, data in enumerate(data_loader):
                data = model.data_preprocessor(data, True)
                inputs = data['inputs']
                data_samples = data['data_samples']
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                with torch.no_grad():
                    model(inputs, data_samples, mode='predict')

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                #print(f"elapsed: {elapsed}")
                if i >= num_warmup:
                    pure_inf_time += elapsed
                    #print(f"pure_inf_time: {pure_inf_time}")
                    if (i + 1) % log_interval == 0:
                        fps = (i + 1 - num_warmup) / pure_inf_time
                        print(f'Done image [{i + 1:<3}/ {total_iters}], '
                              f'fps: {fps:.2f} img / s')

                if (i + 1) == total_iters:
                    fps = (i + 1 - num_warmup) / pure_inf_time
                    print(f'Overall fps: {fps:.2f} img / s\n')
                    benchmark_dict[f'overall_fps_{time_index + 1}'] = round(fps, 2)
                    overall_fps_list.append(fps)
                    break
        benchmark_dict['average_fps'] = round(np.mean(overall_fps_list), 2)
        benchmark_dict['fps_variance'] = round(np.var(overall_fps_list), 4)
        return benchmark_dict

    def print_fps(self, benchmark_dict):
        print_log("\n_____________________________\n"
                  f"FPS {benchmark_dict['average_fps']} img/s +- {benchmark_dict['fps_variance']}"
                  "\n_____________________________\n",
                  logger='current',
                  level=logging.INFO)

    def before_train(self, runner) -> None:
        self.print_fps(self.measure_fps(runner.model.module, runner.val_dataloader))

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs: Optional[dict] = None) -> None:
        if self.every_n_train_iters(runner, self.interval):
            self.print_fps(self.measure_fps(runner.model.module, runner.val_dataloader))

