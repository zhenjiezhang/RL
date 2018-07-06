
#========================================================
# 
# Take multiple iterations of onpolicy aggregation at each iteration refitting the dynamics model to current dataset and then taking onpolicy samples and aggregating to the dataset. 
# Note: You don't need to use a mixing ratio in this assignment for new and old data as described in https://arxiv.org/abs/1708.02596
# 
for itr in range(onpol_iters):
    """ YOUR CODE HERE """
    print(itr, "fitting dyn_model")
    dyn_model.fit(samples)
    
    obs_onpol = []
    next_obs_onpol = []
    act_onpol = []
    rew_onpol = []
    for i in range(num_paths_onpol):
        print("on policy, collecting path", i)
        obs = env.reset()
        obs_onpol.append(obs)
        for step in range(env_horizon):
#             now = time()
#             act, _ = mpc_controller.get_action(obs_onpol[-1])
#             print(time()-now)
            obs, rew, done, _ = env.step(act)
            obs_onpol.append(obs)
            act_onpol.append(act)
            rew_onpol.append(rew)
            
            if done:
                break
                
        next_obs_onpol.extend(obs_onpol[-step:])
        obs_onpol = obs_onpol[:-1]
    
    samples["observations"] = np.vstack([samples["observations"], obs_onpol])
    samples["next_observations"] = np.vstack([samples["next_observations"], next_obs_onpol])
    samples["actions"] = np.vstack([samples["actions"], act_onpol])
    samples["rewards"] = np.vstack([samples["rewards"], rew_onpol])
    
    