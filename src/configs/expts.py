toy_expts = {
    'rrr_toy_expt1': {
        'alg': 'rrr',
        'dataset': 'toy',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'weight_decay': 1e-4,
        'rrr_ap_lamb': 10,
        'rrr_hm_method': 'rrr',
    },
    'rrr_toy_expt2': {
        'alg': 'rrr',
        'dataset': 'toy',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'rrr_ap_lamb': 100,
        'rrr_hm_method': 'rrr',
    },
    'rrr_toy_expt3': {
        'alg': 'rrr',
        'dataset': 'toy',
        'data_frac': -1,
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1,
        'rrr_ap_lamb': 100,
        'rrr_hm_method': 'rrr',
        # 'network_kwargs': {'in_ch': 1, 'in_dim': 2, 'hidden_dim': 200, 'hidden_lay': 3, 'num_cls': 2},
    },
    'ibp_toy_expt1': {
        'alg': 'ibp',
        'dataset': 'toy',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'ibp_EPSILON': 40,
        'ibp_ALPHA': 1,
        'data_frac': -1,
    }
}

mnist_expts = {
    'erm_mnist_expt1': {
        'alg': 'erm',
        'dataset': 'cmnist',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'weight_decay': 1e-4,
        'dataset_kwargs': {'mask_type': 'irrel'},
    },
    'cdep_mnist_expt1': {
        'alg': 'cdep',
        'dataset': 'cmnist',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'weight_decay': 1e-4,
        'dataset_kwargs': {'mask_type': 'relirrel'},
        'cdep_ap_lamb': 1000,
    },
    'ibp_mnist_expt1': {
        'alg': 'ibp',
        'dataset': 'cmnist',
        'user': 'dummy',
        'max_epochs': 50,
        'weight_decay': 1e-4,
        'dataset_kwargs': {'mask_type': 'relirrel'},
        'ibp_EPSILON': .5,
        'ibp_ALPHA': .8,
        'data_frac': 1000,
    }
}

decoy_expts = {
    'erm_dcifar_expt1': {
        'alg': 'erm',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'data_frac': -1
    },
    'ibp_dcifar_expt1': {
        'alg': 'ibp',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'ibp_start_EPSILON': 0.1,
        'ibp_EPSILON': 1,
        'ibp_ALPHA': .95,
        'data_frac': -1
    },
    'ibp_dcifar_expt2': {
        'alg': 'ibp',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'ibp_start_EPSILON': 0,
        'ibp_EPSILON': 2,
        'ibp_ALPHA': 1.,
        'data_frac': -1
    },
    'ibp_dcifar_expt3': {
        'alg': 'ibp',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'ibp_start_EPSILON': 1,
        'ibp_EPSILON': 2,
        'ibp_ALPHA': .9,
        'data_frac': -1
    },
    'ibp_dcifar_expt4': {
        'alg': 'ibp',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'ibp_start_EPSILON': 0,
        'ibp_EPSILON': 3,
        'ibp_ALPHA': 1.,
        'data_frac': -1
    },
    'ibp_rrr_dcifar_expt1': {
        'alg': 'ibp',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'ibp_start_EPSILON': 0,
        'ibp_EPSILON': 1,
        'ibp_ALPHA': 1,
        'ibp_rrr': 1,
        'data_frac': -1
    },
    # 1000 does not work very well.
    'rrr_dcifar_expt1': {
        'alg': 'rrr',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'rrr_ap_lamb': 100,
        'rrr_hm_method': 'rrr',
        'data_frac': -1
    },
    'rrr_dcifar_expt2': {
        'alg': 'rrr',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'rrr_ap_lamb': 10,
        'rrr_hm_method': 'rrr',
        'data_frac': -1
    },
    'rrr_dcifar_expt3': {
        'alg': 'rrr',
        'dataset': 'decoy_cifar10',
        'user': 'dummy',
        'max_epochs': 100,
        'weight_decay': 1e-4,
        'rrr_ap_lamb': 1,
        'rrr_hm_method': 'rrr',
        'data_frac': -1
    },
}

plant_expts = {
    'erm_plant_expt1': {
        'alg': 'erm',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'initialization_factor': 0.1,
    },
    'ibp_plant_expt1': {
        'alg': 'ibp',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 100,
        'ibp_EPSILON': 5e-2,
        'ibp_ALPHA': 1.,
    },
    'ibp_plant_expt2': {
        'alg': 'ibp',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 100,
        'ibp_EPSILON': 1e-2,
        'ibp_ALPHA': 1.,
    },
    # best one
    'ibp_plant_expt3': {
        'alg': 'ibp',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 100,
        'ibp_EPSILON': 1e-1,
        'ibp_ALPHA': 1.,
        'initialization_factor': 0.1,
    },
    'cdep_plant_expt1': {
        'alg': 'cdep',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 100,
        'cdep_ap_lamb': 10,
    },
    'cdep_plant_expt2': {
        'alg': 'cdep',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'cdep_ap_lamb': 100,
        'initialization_factor': 0.1,
    },
    'cdep_plant_expt3': {
        'alg': 'cdep',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'cdep_ap_lamb': 1000,
        'initialization_factor': 0.1,
    },
    'cdep_plant_expt4': {
        'alg': 'cdep',
        'dataset': 'plant',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'cdep_ap_lamb': 10000,
    },
}

expts = {
    'expt1': {
        'alg': 'erm',
        'dataset': 'isic',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'weight_decay': 1e-4
    },
    'expt2': {
        'alg': 'ibp',
        'dataset': 'isic',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'ibp_EPSILON': 1e-1,
        'ibp_ALPHA': 1.,
        'weight_decay': 1e-4
    },
    # best one
    'expt3': {
        'alg': 'ibp',
        'dataset': 'isic',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'ibp_EPSILON': 3e-1,
        'ibp_ALPHA': 1.,
        'weight_decay': 1e-4
    },
    'debug_expt1': {
        'alg': 'ibp',
        'dataset': 'isic',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'ibp_EPSILON': 0,
        'ibp_ALPHA': 1.,
        'weight_decay': 1e-4
    },
    'cdep_isic_expt1': {
        'alg': 'cdep',
        'dataset': 'isic',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'cdep_ap_lamb': 10,
        'weight_decay': 1e-4,
        'batch_size_train': 4,
    },
    'cdep_isic_expt2': {
        'alg': 'cdep',
        'dataset': 'isic',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'cdep_ap_lamb': 100,
        'weight_decay': 1e-4,
        'batch_size_train': 4,
    },
    'cdep_isic_expt3': {
        'alg': 'cdep',
        'dataset': 'isic',
        'user': 'dummy',
        'data_frac': -1,
        'max_epochs': 50,
        'cdep_ap_lamb': 1000,
        'weight_decay': 1e-4,
        'batch_size_train': 4,
    }
}

expts = dict({**expts, **mnist_expts, **plant_expts, **decoy_expts, **toy_expts})
