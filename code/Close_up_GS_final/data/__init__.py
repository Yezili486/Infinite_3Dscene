"""
Data module for Close-up-GS
Contains dataset classes and data loading utilities
Supports LERF and LLFF datasets as specified in paper section 5.1
"""

from .dataset import CloseUpDataset, SyntheticDataset, create_dataloader

__all__ = [
    'CloseUpDataset',
    'SyntheticDataset',
    'create_dataloader',
]