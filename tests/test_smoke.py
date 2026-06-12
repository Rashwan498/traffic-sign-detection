def test_import_core_modules():
    import importlib
    modules = [
        'src.train',
        'src.evaluate',
        'src.data',
    ]
    for m in modules:
        importlib.import_module(m)
