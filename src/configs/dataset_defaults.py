# TODO: num_classes is redundantly defined. It is defined by the dataset.num_classes property, in network_kwargs and
#  num_classes argument
dataset_defaults = {
    'isic': {
        'network_name': 'four_layer_cnn',
        'network_kwargs': {'in_ch': 3, 'in_dim': 299, 'linear_size': 512, 'width': 8, 'num_classes': 2},
        'max_steps': 5000,
        'use_weighted_ce': True,
        'num_classes': 2,
        'num_groups': 3,
        'learning_rate': 3e-4
    },
    'isic_grouped_test': {
        'network_name': 'four_layer_cnn',
        'network_kwargs': {'in_ch': 3, 'in_dim': 299, 'linear_size': 512, 'width': 8, 'num_classes': 2},
        'max_steps': 5000,
        'use_weighted_ce': True,
        'num_classes': 2,
        'learning_rate': 3e-4
    },
    'cmnist': {
        'network_name': 'four_layer_cnn',
        'network_kwargs': {'in_ch': 3, 'in_dim': 28, 'linear_size': 512, 'width': 8, 'num_classes': 10},
        'max_steps': 5000,
        'use_weighted_ce': False,
        'num_classes': 10,
        'num_groups': 1,
        'learning_rate': 3e-4,
        'train_batch_size': 64,
        'test_batch_size': 128
    },
    'plant': {
        'network_name': 'four_layer_cnn',
        'network_kwargs': {'in_ch': 3, 'in_dim': 224, 'linear_size': 1024, 'width': 8, 'num_classes': 2},
        # 'network_name': 'vgg16',
        # 'network_kwargs': {'num_classes': 2},
        'max_steps': 5000,
        'use_weighted_ce': True,
        'num_classes': 2,
        'learning_rate': 3e-5,
        'weight_decay': 1e-5,
        'num_groups': 2,
        'initialization_factor': 0.1,
    },
    'decoy_cifar10': {
        # 'network_name': 'four_layer_cnn',
        # 'network_kwargs': {'in_ch': 3, 'in_dim': 28, 'linear_size': 512, 'width': 8, 'num_classes': 10},
        'network_name': 'ffn',
        'network_kwargs': {'in_ch': 3, 'in_dim': 28*28, 'hidden_dim': 512, 'hidden_lay': 3, 'num_cls': 10},
        'max_steps': 5000,
        'use_weighted_ce': False,
        'num_classes': 10,
        'num_groups': 10,
        'learning_rate': 3e-4,
        'train_batch_size': 64,
        'test_batch_size': 128,
    },
    'toy': {
        'network_name': 'ffn',
        'network_kwargs': {'in_ch': 1, 'in_dim': 2, 'hidden_dim': 50, 'hidden_lay': 3, 'num_cls': 2},
        'max_steps': 5000,
        'use_weighted_ce': False,
        'num_classes': 2,
        'num_groups': 2,
        'learning_rate': 1e-3,
        'train_batch_size': 256,
        'test_batch_size': 256,
    },
}
