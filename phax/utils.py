import pickle


def load(name):
    """load an object using pickle

    Args:
        name: the name to load

    Returns:
        the unpickled object.
    """
    with open(name, "rb") as file:
        obj = pickle.load(file)
    return obj


def save(obj, name):
    """save an object using pickle

    Args:
        obj: the object to save
        name: the name to save the object under
    """
    with open(name, "wb") as file:
        pickle.dump(obj, file)


def validate_params(params):
    """ validate a parameter dictionary

    params: the parameter dictionary. This dictionary should be a possibly
        nested dictionary of floats.
    """
    is_dict_dict = all(isinstance(v, dict) for v in params.values())
    is_float_dict = all(not isinstance(v, dict) for v in params.values())
    msg = "Wrong parameter dictionary format. "
    msg += "Should be a (possibly nested) dictionary of floats or float arrays."
    assert is_float_dict or is_dict_dict, msg


def copy_params(params):
    """copy a parameter dictionary

    Args:
        params: the parameter dictionary to copy

    Returns:
        the copied parameter dictionary

    Note:
        this copy function works recursively on all subdictionaries of the params
        dictionary but does NOT copy any non-dictionary values.
    """
    validate_params(params)
    params = {**params}
    if all(isinstance(v, dict) for v in params.values()):
        return {k: copy_params(params[k]) for k in params}
    return params


def set_global_params(params, **kwargs):
    """add or update the given keyword arguments to each (sub)dictionary of the
       given params dictionary

    Args:
        params: the parameter dictionary to update with the given global parameters
        **kwargs: the global parameters to update the parameter dictionary with.
            These global parameters are often wavelength ('wl') or temperature ('T').

    Returns:
        The modified dictionary.

    Note:
        This operation NEVER updates the given params dictionary inplace.

    Example:
        This is how to change the wavelength to 1600nm for each component in
        the nested parameter dictionary::

            params = set_global_params(params, wl=1.6e-6)
    """
    params = copy_params(params)
    if all(isinstance(v, dict) for v in params.values()):
        return {k: set_global_params(params[k], **kwargs) for k in params}
    params.update(kwargs)
    return params


def get_ports(model):
    """get port names of the model

    Args:
        model: the model dictionary to get the port names from
    """
    ports = {}
    for key in model:
        try:
            p1, p2 = key
            ports[p1] = None
            ports[p2] = None
        except ValueError:
            pass
    return tuple(p for p in ports)


def rename_ports(model, ports):
    """rename the ports of a model

    Args:
        model: the model dictionary to rename the ports for
        ports: a port mapping (dictionary) with keys the old names and values
            the new names.
    """
    original_ports = get_ports(model)
    assert len(ports) == len(original_ports)
    if not isinstance(ports, dict):
        assert len(ports) == len(set(ports))
        ports = {original_ports[i]: port for i, port in enumerate(ports)}
    new_model = {}
    for key in model:
        try:
            p1, p2 = key
            new_model[ports[p1], ports[p2]] = model[p1, p2]
        except ValueError:
            value = model[key]
            if isinstance(value, dict):
                value = {**value}
            new_model[key] = value
    return new_model


def zero(params):
    """the zero model function.

    Args:
        params: the model parameters dictionary

    Returns:
        This function always returns zero.
    """
    return 0.0
