class MockObject(object):
    def __init__(self):
        pass
    def __enter__(self):
        pass
    def __exit__(self,type, value, traceback):
        pass
    def __getattr__(self,name):
        self.name=MockObject()
        return self.name
    
def ConfigProto():
    return MockObject()

def constant(*args, **kwargs):
    return None

def matmul(*args, **kwargs):
    return None

def Session(*args, **kwargs):
    return MockObject()

def device(*args, **kwargs):
    return MockObject()
