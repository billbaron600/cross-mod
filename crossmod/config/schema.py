from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class TaskConfig:
    task_name: str
    seed: int = 0
    input_zip: Optional[str] = None
    output_dir: str = 'run_results'
    keep_image_size: bool = True  # must preserve 1200x1200, never resize
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLConfig:
    algo: str = 'BC'
    steps: int = 100000
    batch_size: int = 128


@dataclass
class ExperimentConfig:
    task: TaskConfig
    rl: RLConfig = field(default_factory=RLConfig)


DEFAULTS: Dict[str, Any] = {
    'task': {
        'seed': 0,
        'output_dir': 'run_results',
        'keep_image_size': True,
    },
    'rl': {
        'algo': 'BC',
        'steps': 100000,
        'batch_size': 128,
    },
}
