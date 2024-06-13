SIM_TIMES = ['small', 'medium', 'large']

MAIN_PARAMS = {  # (tot_num_rounds, eval_every_num_rounds, clients_per_round)
    'synthetic':{
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (1000, 50, 100)
    },
    'sent140': {
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (1000, 50, 100)
    },
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    },
    'fashionmnist': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    },
    'CIFAR': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    },
    'CIFAR100-20': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    },
    'HAR': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    },
    'HARBOX': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    },
    'shakespeare': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (100, 1, 10000)
    }
}

MODEL_PARAMS = {
    # synthetic
    'synthetic.log_reg': (0.003, 10),
    # fashionmnist
    'fashionmnist.CNN': (0.001, 10),
    'fashionmnist.LeNet': (0.05, 10),
    # CIFAR
    'CIFAR.CNN': (0.1, 10),
    'CIFAR100-20.CNN': (0.1, 20),
    # HAR
    'HAR.log_reg': (0.003, 6),
    # HARBOX
    'HARBOX.log_reg': (0.003, 5),
    # shakespeare
    'shakespeare.lstm': (0.0003, 80, 80, 128, 1, 1000), 
    # femnist
    'femnist.cnn': (0.0003, 62, 16384),  # lr, num_classes, max_batch_size
    'femnist.erm_l2': (0.0003, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.erm_log_reg': (0.1, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.rsm_log_reg': (0.1, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.rsm_l2': (0.0003, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.log_reg': (2e-2, 62, round(1e9)),  # lr, num_classes, max_batch_size
    'femnist.erm_cnn_log_reg': (2e-2, 62, 16384),
    # Shakespeare
    # lr, seq_len, num_classes, num_hidden, num_lstm_layers, max_batch_size
    'shakespeare.stacked_lstm': (0.0003, 20, 53, 128, 1, 32768),
    'shakespeare.erm_l2': (0.0003, 20, round(1e9)),
    'shakespeare.rsm_l2': (0.0003, 20, round(1e9)),
    'shakespeare.erm_lstm_log_reg': (0.0003, 20, 53, 128, 1, 32768),
    # Sent 140
    'sent140.erm_log_reg': (0.001, 2, round(1e9)),
    'sent140.rsm_log_reg': (0.1, 2, round(1e9)),  # lr, num_classes, max_batch_size
    'sent140.erm_lstm_log_reg': (0.0003, 2, round(1e9)),  # lr, seq_len, num_classes, max_batch_size
}

MAX_UPDATE_NORM = 100000  # reject all updates larger than this amount

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
AVG_LOSS_KEY = 'avg_loss'

