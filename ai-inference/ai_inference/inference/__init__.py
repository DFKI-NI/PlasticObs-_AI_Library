from importlib import import_module
from inspect import isclass
from pathlib import Path
from pkgutil import iter_modules

import ai_inference.inference as inference
import ai_inference.inference.routes as routes
from ai_inference.inference.routes import ModelInference

# iterate through the inference subpackages
pkg_dir = Path(inference.__file__).resolve().parent
for sub_pkg in Path(pkg_dir).iterdir():
    for module_info in iter_modules([str(sub_pkg)]):
        module_name = module_info.name
        # import the module and iterate through its attributes
        module = import_module(f"ai_inference.{pkg_dir.name}.{sub_pkg.name}.{module_name}")
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isclass(attribute) and issubclass(attribute, ModelInference) and attribute != ModelInference:
                # Add the class to this package's variables
                routes.add_model_inference_class(attribute)
