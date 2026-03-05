"""EMG Data Collection and Preprocessing Package."""

from .collector import EMGCollector, MockSerialGenerator
from .preprocessor import EMGPreprocessor
from .dataset import EMGGestureDataset, create_dataloaders, generate_mock_recordings

__all__ = [
    'EMGCollector',
    'MockSerialGenerator',
    'EMGPreprocessor',
    'EMGGestureDataset',
    'create_dataloaders',
    'generate_mock_recordings',
]
