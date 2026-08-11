from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Any
import numpy as np

class ModelModule(ABC):
    """
    Abstract base class for handling models used by safety cage methods.

    This class defines a common interface for generating predictions, probabilities,
    and intermediate activations from a model. It enables safety cage methods to
    interact with different model types in a consistent way.

    Subclasses must implement methods for prediction, probability estimation, and 
    activation extraction.
    """

    def __init__(
        self,
        selected_layers:Union[str, List[str]],
        use_onehot_encoder:bool,
        model:Any,
        **kwargs
        ):
        """
        Initialize the model module.

        Sets configuration parameters for interacting with the model, such as
        which layers to extract activations from and how outputs are formatted.

        Args:
            model (Any, optional): Underlying model object. (default: None).
            selected_layers (list[str], optional): List of layer names used for feature extraction. (default: None).
            use_onehot_encoder (bool, optional): Whether model outputs are one-hot encoded. (default: False).
            device (str, optional): Device used for model computation (e.g., "cpu", "cuda"). (default: "cpu").
        """
        super(ModelModule, self).__init__()
        
        self.model = model
        self.use_onehot_encoder = use_onehot_encoder

        # Handle different types of selected_layers input
        if isinstance(selected_layers, str):
            self.selected_layers = [selected_layers]
        elif isinstance(selected_layers, list):
            if all(isinstance(layer, str) for layer in selected_layers):
                self.selected_layers = list(selected_layers)
            else:
                raise ValueError("All elements in selected_layers must be strings")
        else:
            raise ValueError(f"selected_layers must be a string or list of strings, got {type(selected_layers)}")


    @abstractmethod
    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get model predictions for input x."""
        raise NotImplementedError("Implement based on your model architecture")
    
    
    @abstractmethod
    def _get_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate activations for each layer given input x."""
        # This method should be implemented based on your specific model architecture
        raise NotImplementedError("Implement based on your model architecture")


    @abstractmethod
    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate pre-activation values for each layer given input x."""
        raise NotImplementedError("Implement based on your model architecture")

if __name__ == '__main__':
    #Quick test to confirm it builds
    model_module = ModelModule()
