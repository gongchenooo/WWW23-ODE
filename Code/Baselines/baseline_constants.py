SIM_TIMES = ['small', 'medium', 'large']

MAIN_PARAMS = {  # (tot_num_rounds, eval_every_num_rounds, clients_per_round)
    'synthetic':{
        'small': (10, 2, 2),
        'medium': (16, 2, 2),
        'large': (1000, 50, 100)
    },
    'CIFAR': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
    'CIFAR100': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
    'CIFAR100-20': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
    'fashionmnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
    'HAR': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
    'HARBOX': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 100)  # TODO Change here constants for experimentation
    },
}

MODEL_PARAMS = {
    # synthetic
    'synthetic.log_reg': (0.0005, 10),
    # CIFAR
    'CIFAR.CNN': (0.1, 10),
    # CIFAR
    'CIFAR100.CNN': (0.1, 100),
    'CIFAR100-20.CNN': (0.1, 20),
    # fashionmnist
    'fashionmnist.CNN': (0.1, 10),
    'fashionmnist.LeNet': (0.05, 10),
    # HAR
    'HAR.log_reg': (0.003, 6),  
    # HARBOX
    'HARBOX.log_reg': (0.003, 5),  
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

