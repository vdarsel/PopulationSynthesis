import argparse

class odict(dict):
    def __init__(self, *args, **kw):
        super(odict,self).__init__(*args, **kw)
    def __getattr__(self, attr):
        return self.get(attr)
    

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace