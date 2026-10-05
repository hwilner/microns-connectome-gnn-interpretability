# Empty conftest: its presence puts the repository root on pytest's
# sys.path (rootdir insertion), so tests can import ``src.attribution``
# without packaging steps or path manipulation.
