"""
Dataclass for the results of a specific type of training
"""

from dataclasses import dataclass, field

from ml.utils.dataclass import SerializableDataclass


@dataclass
class TrainingResults(SerializableDataclass):
    best_validation_loss: float


@dataclass
class FoldResults(SerializableDataclass):
    best_validation_loss: float
    fold_index: int
    n_folds: int


@dataclass
class EnsembleResults(SerializableDataclass):
    folds: list[FoldResults] = field(default_factory=list)


@dataclass
class SystematicResults(SerializableDataclass):
    ensembles: list[EnsembleResults] = field(default_factory=list)


@dataclass
class EstimatorResults(SerializableDataclass):
    systematics: list[SystematicResults] = field(default_factory=list)
