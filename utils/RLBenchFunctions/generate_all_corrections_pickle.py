import pickle
import os

def generate_all_corrections_pickle(config):
    all_corrections = []
    working_dirs = config.working_dirs
    file_to_save_to = os.path.join(config.iteration_working_dir,"all_corrections.pkl")
    # iterate through every folder in working dirs
    # for each one, there is a file called "original_trajectory_and_corrections.pkl"
    # load this pickle in
    # append it to the all_corrections list
    # at the end of the loop, save the file via pickle to the file_to_save_to path
    for dir_path in working_dirs:
        file_path = os.path.join(dir_path, "original_trajectory_and_corrections.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                all_corrections.append(data)
        else:
            print(f"Warning: {file_path} not found. Skipping.")
    
    with open(file_to_save_to, 'wb') as f:
        pickle.dump(all_corrections, f)