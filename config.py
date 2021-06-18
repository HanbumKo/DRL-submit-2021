import torch



# model parameters
device = torch.device("cuda:1") if torch.cuda.is_available else torch.device("cpu")
hidden_dim = 256

# TSP paprameters
n_nodes = 20

# Env parameters
n_rows = 4

# Hyperparameters
total_step = 60000000
test_batch_size = 256
gamma = 0.99

# MSC
random_seed = 0


actor_weight_path = "./DFPG_A2C_hybrid/n_" + str(n_nodes) + "/best_model_actor_truck_params_n_" + str(n_nodes) + "_DFPG.pkl"
critic_weight_path = "./DFPG_A2C_hybrid/n_" + str(n_nodes) + "/best_model_critic_params_n_" + str(n_nodes) + "_DFPG.pkl"