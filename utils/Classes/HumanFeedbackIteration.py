import os
import pickle 

class HumanFeedbackIteration:
    def __init__(self, current_policy_trajectories, user_changes, working_dir, working_dirs):
        self.current_policy_trajectories = current_policy_trajectories
        self.user_changes = user_changes
        self.working_dir = working_dir
        self.working_dirs = working_dirs

    def save_object(self):
        # Specify the file path where the object will be saved
        file_path = os.path.join(self.working_dir, "human_feedback_iteration.pkl")
        
        # Create the working directory if it doesn't exist
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Pickle the object
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        
        print(f"HumanFeedbackIteration object saved to {file_path}")