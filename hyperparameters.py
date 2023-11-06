dataset_path = "./CTP_Wires_Chargers_etc"
batch_size = 24
num_workers = 16 # number of cores/processes on CPU to load data
epochs = 2
crop_size = 84

learning_rate_init = 2e-3
lam = 0.5
eta = 1.2
decay_rate = 0.9

I_max = 20 # max steps to train the network
L_max = 1.8
L_min = 1.2

checkpoint_epoch = 10

sample_trial = 5

debug = True