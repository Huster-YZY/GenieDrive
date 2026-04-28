from .pipeline_wan import WanPipeline

from .pipeline_wan_fun_control import WanFunControlPipeline

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)
    WanFunControlPipeline.__call__ = sparse_reset(WanFunControlPipeline.__call__)