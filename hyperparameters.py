dataset_path = "./CTP_Wires_Chargers_etc"
batch_size = 24
num_workers = 8 # number of cores/processes on CPU to load data
epochs = 10
crop_size = 64

learning_rate_init = 2e-3
lam = 0.5
eta = 1.2
decay_rate = 0.9

sample_trial = 5