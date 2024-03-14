# Exploiting Inter-Sample Information for Long-tailed Out-of-Distribution Detection 

This is the official implementation of the [Exploiting Inter-Sample Information for Long-tailed Out-of-Distribution Detection]().


## Package installation
* python                    3.11.4
* pytorch-cuda              11.8 
* torchvision               0.15.2
* torch-cluster             1.6.1+pt20cu118
* torch-geometric           2.3.1

## Train and test

CIFAR10-LT: 

```
python cifar_LT_GCN.py --ds cifar10  --drp <path_to_input_datasets> --op <path_to_output_files>
```

CIFAR100-LT:

```
python cifar_LT_GCN.py --ds cifar100  --drp <path_to_input_datasets> --op <path_to_output_files>
```

ImageNet-LT:

```
python imageNet_LT_GCN.py  --drp <path_to_input_datasets> --op <path_to_output_files>
```


## Pretrained models

Pretrained models are available on [Google Drive]()


## Acknowledgement

Part of our codes are adapted from these repos:

long-tailed-ood-detection - https://github.com/amazon-science/long-tailed-ood-detection - Apache-2.0 license

pre-training - https://github.com/hendrycks/pre-training/ - Apache-2.0 license


## Citation
```

```

## Security



## License

This project is licensed under the '' License.
