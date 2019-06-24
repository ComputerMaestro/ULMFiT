using DataDeps
using HDF5

register(DataDep(
    "pretrained-ULMFiT",
    """
    This is pretrained ULMFiT model, the weights were provided by the authors Jeremy Howard and Sebastian Ruder on official fastai's site.
    http://files.fast.ai/models/wt103/
    """,
))
