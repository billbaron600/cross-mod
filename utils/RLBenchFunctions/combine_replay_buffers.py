from utils.Classes.custom_replay_buffer import ScaledRewardBuffer
from utils.RLBenchFunctions.fill_buffer import preload_from_config
from utils.Classes.policy_trajectory_class import PolicyTrajectory

def _first_env(arr):
    """
    Return env-0 while *preserving* the env axis so shapes stay compatible.
    Handles 2-D (N, n_envs) and >=3-D (N, n_envs, …) arrays.
    """
    if arr.ndim >= 2 and arr.shape[1] > 1:
        if arr.ndim>2:
            return arr[:, 0:1, ...]          # (N, 1)  or  (N, 1, …)
        else:
            return arr[:,0:1]
    return arr                            # already single-env

def combine_replay_buffers(config,model,buffer_paths,buffer_size=3000000,create_demo_buffer=False,db_index=400,include_non_expert=False):
    #concatentaet raw_rewards, raw_rewards_unclipped, and infos
    #combine is_demo. This will be a littler harder, but still straightforward
    #remove the middle dimension of actions, (Nx8 numpy array after), and concatentate them
    #Iterate through the keys in replay_buffer.observations, remove the middle dimensions, and concatetnate
    #Run recompute stats on the updated buffer
    #update pos

    if create_demo_buffer is True:
        model.buffer_size = buffer_size
        buffer_kwargs = {
            "buffer_size":buffer_size,
            "demo_fraction": 0.2,
            "reward_shift": 0.0,
            "adjust_mean_every": 500,
            "success_bonus": 5.0,
            "restrict_recent": True,
            "recent_window": 50_000,
            "distribute_steps": 1,
            "clip_limit": (-10.0, 10.0),
            "raw_clip_limit": (-6.0, 10.0),
            "observation_space":model.observation_space,
            "action_space":model.action_space}
        
        model.replay_buffer = ScaledRewardBuffer(**buffer_kwargs)
        model = preload_from_config(model, config,db_index=db_index)
        buffer_path_idx = 0
    else:
        model.load_replay_buffer(buffer_paths[0])
        buffer_path_idx=1
    
    for buffer_idx in range(buffer_path_idx, len(buffer_paths)):
        old_buf  = model.replay_buffer                      # buffer already holding data
        old_pos  = old_buf.pos                              # current write-pointer

        # --- load the next file into *model.replay_buffer* (temporary) ---
        model.load_replay_buffer(buffer_paths[buffer_idx])
        new_buf  = model.replay_buffer
        new_len  = new_buf.pos                              # how many valid rows

        # make sure the target buffer is large enough
        new_total = old_pos + new_len
        if new_total > old_buf.buffer_size:
            raise ValueError(f"Not enough capacity ({old_buf.buffer_size}) "
                            f"for {new_total} transitions")

        # ----------------------------------------------------------------
        # 1)  ACTIONS  (collapse env axis → copy)
        old_buf.actions[old_pos:new_total] = _first_env(
            new_buf.actions[:new_len]
        )

        # OBS / NEXT_OBS
        for key in old_buf.observations:
            old_buf.observations[key][old_pos:new_total]       = _first_env(
                new_buf.observations[key][:new_len]
            )
            old_buf.next_observations[key][old_pos:new_total]  = _first_env(
                new_buf.next_observations[key][:new_len]
            )

        # REWARDS (now matches (N, 1))
        old_buf.rewards[old_pos:new_total]     = _first_env(new_buf.rewards[:new_len])
        old_buf.raw_rewards[old_pos:new_total] = _first_env(new_buf.raw_rewards[:new_len])

        # DONES
        old_buf.dones[old_pos:new_total]       = _first_env(new_buf.dones[:new_len])

        # IS_DEMO is already 1-D in both buffers; no change needed
        old_buf.is_demo[old_pos:new_total]     = new_buf.is_demo[:new_len]

        # INFOS list/array copy stays the same
        old_buf.infos.extend(new_buf.infos)
        # ----------------------------------------------------------------
        # advance write-pointer and refresh stats
        old_buf.pos = new_total
        old_buf.recompute_stats()                 # your custom routine

        model.replay_buffer = old_buf
    
    



    return model