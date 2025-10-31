import re, torch, contextlib
from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
)

from transformers.generation.utils import GenerationMixin
from utils.malmo_query import collect_molmo_points, plot_views_with_prompts
import pickle
import os

#Specify key info
iteration_working_dir = "run_results/rubish_in_bin/demos"
#limit_to_correction_indices = [0,1,2,3] #,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
limit_to_correction_indices = [20] #10,11,12,13,14,15,16,17,18,19,20]
n_views = [0,1]

#seed vie probs: 0,1,3,9

generate_points = True
use_first_prompt = False #only use the first views prompts, rather than all of them
cropped = False
plot = False



prompts = None #[view0_prompts, view1_prompts]


# ── assemble kwargs and call helper ─────────────────────────────────
kwargs = {
    "iteration_working_dir": iteration_working_dir,
    "seeds":                 limit_to_correction_indices,
    "n_views":               n_views,
    "prompts": prompts,
}

if generate_points is True:
    results = collect_molmo_points(cropped=cropped,use_first_prompt=use_first_prompt,**kwargs)

if plot is True:
    path_use = os.path.join(iteration_working_dir,str(limit_to_correction_indices[0]))
    plot_views_with_prompts(iteration_working_dir=path_use)

