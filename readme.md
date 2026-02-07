# PytorchDevUtils 🧰

**Colletion of utilities for debugging and developing Pytorch models. Print shape hooks, handy logger and metrics.** 
> Note: This is not and will never be a production ready library, but a collection of utilities that fits my needs. Probably there are more complex libraries for each of the tools here.

## Tools 🛠️

- [Shape Hooks 🪝](#shape-hooks-🪝)
- [Logger 📜](#logger-📜)
- ... [WIP]

### Shape Hooks 🪝

This is a utility to print the shapes of the tensors in input and in output of all the modules in a Pytorch model. It works with the hooks (forward-hook) but they are disabled after the first usage, to avoid log-hell. It can be used as a sort of debug for shapes of tensors even when you're not debugging.  

```python
model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 16 * 16, 10))
hook_manager = ShapeHook()
hook_manager.register_hooks(model, one_time=True)
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)
```
The output will be:

```txt
ShapeHook for Conv2d          in shapes: [[1, 3, 32, 32]]              out shape: [1, 32, 32, 32]               
ShapeHook for ReLU            in shapes: [[1, 32, 32, 32]]             out shape: [1, 32, 32, 32]               
ShapeHook for MaxPool2d       in shapes: [[1, 32, 32, 32]]             out shape: [1, 32, 16, 16]               
ShapeHook for Flatten         in shapes: [[1, 32, 16, 16]]             out shape: [1, 8192]                     
ShapeHook for Linear          in shapes: [[1, 8192]]                   out shape: [1, 10]    
```


### Logger 📜

This utility handles logging functionality. It also serves as a wrapper for wandb, if enabled. By utilizing log levels, it allows logging only critical information to the console while logging all details to wandb. Future updates may include support for Telegram notifications and additional logging integrations.

```python
model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 16 * 16, 10))
logger = Logger(print_log_level=Logger.LogLevel.INFO)
# logger.initialize_wandb("test_project")
logger.log({"message": "Starting model forward pass"}, log_level=Logger.LogLevel.DEBUG)
        
dummy_input = torch.randn(1, 3, 32, 32)
output = model(dummy_input)

logger.log({"message": "Model forward pass completed"}, log_level=Logger.LogLevel.INFO)

```

```txt
[INFO] 2025-04-19 12:52:17: {
    "message": "Model forward pass completed"
}
```

## Installation 🚀

Code is not yet available on PyPI. You can install it from source:

```bash
git clone https://github.com/jacksalici/PytorchDevUtils
cd PytorchDevUtils
source activate your_env 
pip install -e . # use -e for editable install, if you want the code to be updated automatically
```

## License
MIT
