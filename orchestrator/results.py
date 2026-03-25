"""
Dataclass for the results of a specific type of training
"""

from dataclasses import dataclass, field, asdict 

from ml.utils.dataclass import SerializableDataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import json


@dataclass
class TrainingResults(SerializableDataclass):
    best_validation_loss: float


@dataclass
class FoldResults(SerializableDataclass):
    best_validation_loss: float
    fold_index: int
    n_folds: int

    #def __post_init__(self):
    #    """Ensure folds are FoldResults instances"""
    #    self.folds = [
    #        FoldResults(**f) if isinstance(f, dict) else f
    #        for f in self.folds
    #    ]


@dataclass
class EnsembleResults(SerializableDataclass):
    folds: list[FoldResults] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str):
        """Override to properly deserialize nested FoldResults"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert dict folds to FoldResults objects
        if 'folds' in data:
            data['folds'] = [
                FoldResults(**fold_dict) if isinstance(fold_dict, dict) else fold_dict
                for fold_dict in data['folds']
            ]
        
        return cls(**data)


@dataclass
class SystematicResults(SerializableDataclass):
    ensembles: list[EnsembleResults] = field(default_factory=list)


@dataclass
class EstimatorResults(SerializableDataclass):
    systematics: list[SystematicResults] = field(default_factory=list)




class AggregationMethod(str, Enum):
    """Enum for aggregation methods - inherit from str for JSON serialization"""

    BEST = "best"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    VOTING = "voting"
    SUM = "sum"


@dataclass
class ModelNodeMetadata:
    """Represents a single trained model in the DAG"""

    checkpoint_path: str
    task_type: Literal["fold", "ensemble", "systematic", "estimator", "main"]
    fold_index: Optional[int] = None
    ensemble_index: Optional[int] = None
    estimator_name: Optional[str] = None
    systematic_name: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class AggregationEdge:
    """Defines how child nodes aggregate into parent"""

    method: AggregationMethod
    source_nodes: List[str]  # node IDs
    target_node: str
    weights: Optional[List[float]] = None  # for weighted_mean
    metric_key: Optional[str] = None  # for "best" selection


@dataclass
class DAGSnapshot:
    """Complete snapshot of the trained model ensemble"""

    nodes: Dict[str, ModelNodeMetadata]  # node_id -> metadata
    edges: List[AggregationEdge]
    config_snapshot: Dict[str, Any]  # original Hydra config
    root_node: str  # entry point for evaluation

    def to_json(self, path: str):
        """Serialize to JSON with proper enum handling"""
        with open(path, "w") as f:
            json.dump(
                {
                    "nodes": {k: asdict(v) for k, v in self.nodes.items()},
                    "edges": [asdict(e) for e in self.edges],
                    "config_snapshot": self.config_snapshot,
                    "root_node": self.root_node,
                },
                f,
                indent=2,
            )

    @classmethod
    def from_json(cls, path: str):
        """Deserialize from JSON"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            nodes={k: ModelNodeMetadata(**v) for k, v in data["nodes"].items()},
            edges=[
                AggregationEdge(
                    method=AggregationMethod(e["method"]),
                    source_nodes=e["source_nodes"],
                    target_node=e["target_node"],
                    weights=e.get("weights"),
                    metric_key=e.get("metric_key"),
                )
                for e in data["edges"]
            ],
            config_snapshot=data["config_snapshot"],
            root_node=data["root_node"],
        )