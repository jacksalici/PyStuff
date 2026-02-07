import torch
import torch.nn as nn
from typing import Dict

class ShapeHook:
    """
    Singleton class to manage PyTorch hooks that print tensor shapes.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ShapeHook, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self._hooks: Dict[str, Dict] = {}
        self._one_time_mode = False
    
    def register_hooks(self, model: nn.Module, one_time: bool = False) -> None:
        """
        Register hooks on all modules of a model to print tensor shapes.
        
        Args:
            model (nn.Module): PyTorch model to register hooks on
            one_time (bool): If True, hooks will be removed after they fire once
        """
        self._one_time_mode = one_time
        model_id = id(model)
        
        if model_id in self._hooks:
            self.remove_model_hooks(model)
        
        self._hooks[model_id] = {} # model hooks dictionary
        
        for name, module in model.named_modules():
            if name == '': #skip the root module
                continue
            
            def create_hook_fn(module_name, module_id):
                def hook_fn(module, input, output):
                    # print the input shape
                    print("ShapeHook for ", end="")
                    if isinstance(input, torch.Tensor):  
                        print(f"{module_name: <15} in shape: {str(list(input.shape)): <30}", end="")
                    elif isinstance(input, tuple) and all(isinstance(i, torch.Tensor) for i in input):
                        shapes = [list(i.shape) for i in input]
                        print(f"{module_name: <15} in shapes: {str(shapes): <30}", end="")
                    else:
                        print(f"{module_name: <15} in type: {str(type(input)): <30}", end="")
                    
                    
                    if isinstance(output, torch.Tensor):
                        print(f"out shape: {str(list(output.shape)): <30}")
                    elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
                        shapes = [list(o.shape) for o in output]
                        print(f"out shapes: {str(shapes): <30}")
                    else:
                        print(f"out type: {str(type(output)): <30}")
                    
                    # if one_time, remove the hook on the module after first call
                    if self._one_time_mode and model_id in self._hooks and module_id in self._hooks[model_id]:
                        self._hooks[model_id][module_id].remove()
                        del self._hooks[model_id][module_id]
                
                return hook_fn
            
            module_id = id(module)
            hook_fn = create_hook_fn(module.__class__.__name__, module_id)
            handle = module.register_forward_hook(hook_fn)
            # store the handle in the hooks dictionary for later removal
            self._hooks[model_id][module_id] = handle
    
    def remove_model_hooks(self, model: nn.Module) -> None:
        model_id = id(model)
        if model_id in self._hooks:
            for module_id, handle in list(self._hooks[model_id].items()):
                handle.remove()
            self._hooks[model_id].clear()
    
    def remove_all_models_hooks(self) -> None:
        """Remove all hooks from all models."""
        for model_id in list(self._hooks.keys()):
            for module_id, handle in list(self._hooks[model_id].items()):
                handle.remove()
            self._hooks[model_id].clear()


# Example usage
if __name__ == "__main__":
    # Create a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 10)
    )
    
    hook_manager = ShapeHook()
    
    hook_manager.register_hooks(model, one_time=True)
    
    print("First forward pass (will print all shapes):")
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    
    print("\nSecond forward pass (should not print shapes):")
    output = model(dummy_input)
    
    print("\nEnd.")
    