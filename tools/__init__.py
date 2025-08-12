# from .runner import run_net
from .runner import test_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_net as finetune_test_run_net
from .runner_cache_prompt import run_net as cp_run_net
from .runner_point_prompt import run_net as prompt_run_net
from .runner_module import run_net as module_run_net
from .runner_module import test_net as module_tune_test_run_net
from .runner_pretask import run_net as pretask_run_net
from .runner_finetune_seg import run_net as finetune_seg_run_net
from .runner_unify_seg import run_net as unify_seg_run_net
from .runner_finetune_ensemble import run_net as ensemble_run_net