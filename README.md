# Tight PAC-Bayes Compression Bounds

[![](https://img.shields.io/badge/arXiv-2211.xxxxx-red)]() [![](https://img.shields.io/badge/NeurIPS-2022-green)]()

This repository hosts the code for [PAC-Bayes Compression Bounds So Tight That They Can Explain Generalization]() by [Sanae Lotfi*](https://sanaelotfi.github.io), [Marc Finzi*](https://mfinzi.github.io), [Sanyam Kapoor*](https://sanyamkapoor.com), [Andres Potapczynski*](https://www.andpotap.com), [Micah Goldblum](https://goldblum.github.io), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).


### Training Intrinsic Dimensionality Models


```shell
python experiments/train.py --dataset=cifar10 \
                            --model-name=Layer13s \
                            --base-width=8 \
                            --optimizer=adam \
                            --epochs=500 \
                            --lr=1e-3 \
                            --intrinsic_dim=3500 \
                            --intrinsic_mode=filmrdkron \
                            --seed=137 \
                            --is_said=True \
                            --is_layered=False
```

All arguments in the `main` method of [experiments/train.py](./experiments/train.py)
are valid CLI arguments. The most imporant ones are noted here:

* `--seed`: Setting the seed is important so that any subsequent runs using the checkpoint can reconstruct the same random parameter projection matrices used during training.
* `--data_dir`: Parent path to directory containing root directory of the dataset.
* `--dataset`: Dataset name. See [data.py](./pactl/data.py) for list of dataset strings.
* `--intrinsic_dim`: Dimension of the training subspace of parameters.
* `--intrinsic_mode`: Method used to generate (sparse) random projection matrices. See `create_intrinsic_model` method in [projectors.py](./pactl/nn/projectors.py) for a list of valid modes.



### Computing our Adaptive Compression Bounds

Once we have the checkpoints of intrinsic-dimensionality models, the bound can be computed using:

```shell
python experiments/compute_bound.py --dataset=cifar10 \
                                    --misc-extra-bits=7 \
                                    --quant-epochs=30 \
                                    --levels=50 \
                                    --lr=0.0001 \
                                    --prenet_cfg_path=/nfs/scistore23/chlgrp/dghobadi/tight-pac-bayes/wandb/offline-run-20240913_103551-6ocsrtt5/files/net.cfg.yml \
                                    --use_kmeans=True \
				     --is_said=False \
                                    --quantize_type=default \
                           	     --said_quantize_type=float8
```

The key arguments here are:
* `--misc-extra-bit`: Penalty for hyper-parameter optimization during bound computation, equals the bits required to encode all hyper-parameter configurations.
* `--levels`: Number of quantization levels.
* `--quant-epochs`: Number of epochs used for fine-tuning of quantization levels.
* `--lr`: Learning rate used for fine-tuning of quantization levels.
* `--user_kmeans`: When true, uses kMeans clustering for initialization of quantization levels. Otherwise, random initialization is used.

## LICENSE

Apache 2.0
