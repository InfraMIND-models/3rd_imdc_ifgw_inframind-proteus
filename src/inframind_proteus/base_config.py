from dataclasses import dataclass, fields
from typing import Any, Union


@dataclass
class BaseConfig:
    """Base class for nested configuration objects.

    Useful for representing data hierarchically (JSON/YAML-like) with
    structured fields.

    Similar to a pydantic BaseModel, but for this project we want to
    keep it simple and use our own.
    """
    #
    # def __init_subclass__(cls, **kwargs):
    #     """Automatically applies the dataclass decorator to all subclasses."""
    #     super().__init_subclass__(**kwargs)
    #     dataclass(cls)

    def update_from_dict(self, config_dict: Union[dict[str, Any], None]) -> None:
        """Update the configuration from a dictionary.
        New fields are added, regardless of whether they exist in the current
        config or not.
        If a field matches another config object, the update is applied
        recursively to that config object.
        """
        config_dict = config_dict or {}
        for field_name, field_value in config_dict.items():
            is_cfg = False
            if hasattr(self, field_name):
                attr = getattr(self, field_name)
                if issubclass(attr.__class__, BaseConfig):
                    attr.update_from_dict(field_value)
                    is_cfg = True

            if not is_cfg:
                setattr(self, field_name, field_value)

            if isinstance(field_value, BaseConfig):
                field_value.update_from_dict(config_dict.get(field_name, {}))

    def preprocess(self, *args, **kwargs):
        """Note: Call super when overriding."""
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, BaseConfig):
                field_value.preprocess(*args, **kwargs)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], preprocess=True) -> "BaseConfig":
        """Create a config object from a dictionary."""
        cfg = cls()
        cfg.update_from_dict(config_dict)
        if preprocess:
            cfg.preprocess()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Convert the config object to a dictionary, recursing through
        nested config members.

        Data is shallow-copied, so mutable objects (e.g. lists) will be the
        same as the one held by this object.

        This function does not apply any special YAML-friendly serialization,
        but you can call `make_yaml_exportable_dict()` after this before
        exporting to YAML.
        """

        # Select non-callable and non-private class attributes
        d = {
            key: value for key, value in self.__class__.__dict__.items()
            if not key.startswith("__") and not callable(value)
        }
        # Update with instance attributes, which may override class attributes
        d.update(self.__dict__)

        # Recurse into nested BaseConfig objects
        for field_name, field_value in d.items():
            if isinstance(field_value, BaseConfig) or hasattr(field_value, "to_dict"):
                d[field_name] = field_value.to_dict()

        return d

        # # Alternative approach: Works if all subclasses are dataclasses.
        # result = {}
        # for f in fields(self):
        #     value = getattr(self, f.name)
        #     if isinstance(value, BaseConfig):
        #         result[f.name] = value.to_dict()
        #     else:
        #         result[f.name] = value
        #
        # return result
