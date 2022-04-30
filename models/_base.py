import copy

_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls

    return decorator


def get_model(cfg):
    m_dict = copy.deepcopy(cfg)
    model_type = m_dict.pop('type')
    return _MODEL_DICT[model_type](**m_dict)
