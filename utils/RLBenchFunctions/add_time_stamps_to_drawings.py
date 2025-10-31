import torch

def add_time_stamps_to_drawings(grand_traj_tor_r):
    all_view_list=[]
    for view in range(len(grand_traj_tor_r)):
        current_trajs=grand_traj_tor_r[view]
        view_list=[]
        for traj_ind in range(len(current_trajs)):
            c_traj=current_trajs[traj_ind][0]
            cur_time=(torch.arange(len(c_traj))*(1/len(c_traj)))[:,None]
            traj_wt=torch.cat([cur_time,c_traj],dim=-1)
            view_list.append(traj_wt.clone())
            view_tor=torch.vstack(view_list)
        all_view_list.append(view_tor.clone())
    return all_view_list