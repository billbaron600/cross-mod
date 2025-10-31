import json, sys, pathlib
from utils.malmo_query import collect_molmo_points

with open(sys.argv[1]) as fh:
    kwargs = json.load(fh)

results = collect_molmo_points(**kwargs)

# optional: stash results somewhere useable by the caller
out_file = pathlib.Path(kwargs["iteration_working_dir"]) / "points.pkl"
with open(out_file, "wb") as fh:
    import pickle; pickle.dump(results, fh)