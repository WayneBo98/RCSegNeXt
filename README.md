# RCSegNeXt
This is an official implementation of our MIDL 2025 subumision work: 'RCSegNeXt: Efficient multi-scale ConvNeXt for rectal cancer segmentation from sagittal MRI scans'.

This repository is based on the framework nnUNetv2, many thanks to their excellent work.

The network architectures are in nnunetv2/Network, and the corresponding trainer for the architectures are in the nnunetv2/training/nnUNetTrainer.

One can use 'bash train.sh' to train their own dataset (use nnUNetv2's preprocessing first), trainer can be changed to other names for corresponding architectures.

Official checkpoints will be avaliable very soon.
