from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

@dataclass
class IterateConfig:
    positions_per_iteration: int = 1
    sampling_method: str = "top_p"
    top_p: float = 1
    # top_p: float = 0.75
    top_k: int = 42
    # sampling_temperature: float = 1
    sampling_temperature: float = 2

    # maksing
    position_selection: str = "confidence"
    unmask_tradeoff_alpha= 0.0

@dataclass
class RefineConfig:
    confidences_min_threshold: float = 0.9
    initial_mask_ratio: float = 0.75
    mask_ratio_rate: float = 0.001
    min_mask_ratio: float = 0.01
    max_refine_rounds: int = 10000
    max_score = 0.95

    position_selection= "confidence"   # 'entropy'|'confidence'
    random_strategy    = "epsilon"      # "epsilon" | "gumbel" | "random" | "greedy"
    epsilon            = 0.5           # random_strategy == "epsilon"
    # epsilon            = 0.2           # random_strategy == "epsilon"
    gumbel_tau         = 1.0     

    # sampling
    positions_per_iteration: int = 2
    sampling_method: str = "top_p"             # 'greedy'|'sampling'|'top_k'|'top_p'
    top_k: int = 42
    top_p: float = 0.9
    # top_p: float = 0.6
    sampling_temperature: float = 1.5
    # sampling_temperature: float = 1.0
    unmask_tradeoff_alpha = 0



@dataclass
class MCTSConfig:
    iterations: int = 5000
    max_depth: int = 80
    c_puct: float = 1
    K_pos: int = 28
    K_base: int = 4
    top_pos_by: str = "confidence"                # 'entropy'|'confidence'
    add_root_dirichlet: bool = False
    tree_log_every: int = 100
    tree_debug: bool = False

@dataclass
class RTGenerationConfig:
    # core
    rna_length: int = 28
    device: str = "cuda"
    seed: int = 42
    iterate: IterateConfig = field(default_factory=IterateConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)

@dataclass
class GenerationResult:
    best_sequence: str
    best_value: float
    all_candidates: List[str] = field(default_factory=list)
    init_sequence: Optional[str] = None
    scores_history: List[Tuple[int, float]] = field(default_factory=list)
    best_scores: dict = None
