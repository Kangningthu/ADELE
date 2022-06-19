import importlib
try:
    from comet_ml import Experiment as CometExperiment
    from comet_ml import OfflineExperiment as CometOfflineExperiment
except ImportError:  # pragma: no-cover
    _COMET_AVAILABLE = False
else:
    _COMET_AVAILABLE = True


import torch
from torch import is_tensor
from typing import Any, Dict, Optional, Union
from datetime import datetime

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

class CometWriter:
    def __init__(
        self,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        api_key: Optional[str] = None, 
        log_dir: Optional[str] = None, 
        offline: bool = False,
        **kwargs):
        if not _COMET_AVAILABLE:
            raise ImportError(
                "You want to use `comet_ml` logger which is not installed yet,"
                " install it with `pip install comet-ml`."
            )

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.kwargs = kwargs

        self.timer = Timer()


        if (api_key is not None) and (log_dir is not None):
            self.mode = "offline" if offline else "online"
            self.api_key = api_key
            self.log_dir = log_dir

        elif api_key is not None:
            self.mode = "online"
            self.api_key = api_key
            self.log_dir = None
        elif log_dir is not None:
            self.mode = "offline"
            self.log_dir = log_dir
        else:
            print("CometLogger requires either api_key or save_dir during initialization.")

        if self.mode == "online":
            self.experiment = CometExperiment(
                api_key=self.api_key,
                project_name = self.project_name,
                **self.kwargs,
            )
        else:
            self.experiment = CometOfflineExperiment(
                offline_directory=self.log_dir,
                project_name=self.project_name,
                **self.kwargs,
            )

        if self.experiment_name:
            self.experiment.set_name(self.experiment_name)

    def set_step(self, step, epoch = None, mode='train') -> None:
        self.mode = mode
        self.step = step
        self.epoch = epoch
        if step == 0:
            self.timer.reset()
        else:
            duration = self.timer.check()
            self.add_scalar({'steps_per_sec': 1 / duration})

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        self.experiment.log_parameters(params)
    
    def log_code(self, file_name = None, folder = 'models/') -> None:
        self.experiment.log_code(file_name=file_name, folder=folder)


    def add_scalar(self, metrics: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None, epoch: Optional[int] = None) -> None:
        metrics_renamed = {}
        for key, val in metrics.items():
            tag = '{}/{}'.format(key, self.mode)
            if is_tensor(val):
                metrics_renamed[tag] = val.cpu().detach()
            else:
                metrics_renamed[tag] = val
        if epoch is None and step is None:
            self.experiment.log_metrics(metrics_renamed, step = self.step, epoch = self.epoch)
        elif epoch is None and step is not None:
            self.experiment.log_metrics(metrics_renamed, step = step)
        elif epoch is not None and step is None:
            self.experiment.log_metrics(metrics_renamed, epoch = epoch)
        else:
            self.experiment.log_metrics(metrics_renamed, step = step, epoch = epoch)

    def add_plot(self, figure_name, figure):
        """
        Primarily for log gate plots
        """
        self.experiment.log_figure(figure_name = figure_name, figure = figure)

    def add_text(self, text, step):
        """
        Primarily for log gate plots
        """
        self.experiment.log_text(text, step = step)

    def add_hist3d(self, hist, name):
        """
        Primarily for log gate plots
        """
        self.experiment.log_histogram_3d(hist, name = name)

    def reset_experiment(self):
        self.experiment = None

    def finalize(self) -> None:
        self.experiment.end()
        self.reset_experiment()
           