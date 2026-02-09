"""Storage adapters for reading Optuna results."""

from __future__ import annotations

from typing import Protocol

from collections.abc import Iterable, Sequence
from pathlib import Path
import re

import optuna

from .config import ScenarioConfig
from .layout import Role, runs_dir
from .toolkit.logging import get_logger

_logger = get_logger(__name__)


class TrialStore(Protocol):
    """Protocol for loading and pairing trial results."""

    def load_best_params(self, db_path: Path, study_name: str | None) -> dict[str, float] | None:
        """Load best parameter set from a DB/study."""

    def list_studies(self, db_path: Path) -> list[str]:
        """List study names in a DB."""

    def pair_trials(
        self,
        scenario: ScenarioConfig,
        role: Role,
        db_paths: Sequence[Path],
    ) -> list[tuple[int, dict[str, float], dict[str, float]]]:
        """Pair macro/micro best params by run_id."""


class OptunaTrialStore:
    """Optuna-backed TrialStore implementation."""

    def load_best_params(self, db_path: Path, study_name: str | None) -> dict[str, float] | None:
        """Load best trial parameters for a DB and study.

        Args:
            db_path: Path to sqlite Optuna DB.
            study_name: Optional study name. If `None`, infer from DB.

        Returns:
            dict[str, float] | None: Best parameter mapping when available.
        """
        storage_uri = f"sqlite:///{db_path.resolve()}"
        try:
            if study_name is None:
                studies = optuna.get_all_study_summaries(storage=storage_uri)
                if len(studies) != 1:
                    _logger.debug("Expected one study in %s but found %d", db_path, len(studies))
                    return None
                study_name = studies[0].study_name

            study = optuna.load_study(study_name=study_name, storage=storage_uri)
            if study.best_trial:
                return {name: float(value) for name, value in study.best_trial.params.items()}
        except Exception as exc:
            _logger.debug("Failed to load study %s from %s: %s", study_name, db_path, exc)
            return None
        return None

    def list_studies(self, db_path: Path) -> list[str]:
        """List study names in an Optuna DB.

        Args:
            db_path: Path to sqlite Optuna DB.

        Returns:
            list[str]: Study name list.
        """
        storage_uri = f"sqlite:///{db_path.resolve()}"
        try:
            studies = optuna.get_all_study_summaries(storage=storage_uri)
            return [study.study_name for study in studies]
        except Exception as exc:
            _logger.debug("Failed to list studies from %s: %s", db_path, exc)
            return []

    def pair_trials(
        self,
        scenario: ScenarioConfig,
        role: Role,
        db_paths: Sequence[Path],
    ) -> list[tuple[int, dict[str, float], dict[str, float]]]:
        """Pair macro/micro best params by run id.

        Args:
            scenario: Scenario definition.
            role: Target role (`train` or `eval`).
            db_paths: Candidate DB file paths.

        Returns:
            list[tuple[int, dict[str, float], dict[str, float]]]: Sorted paired parameter list.
        """
        pairs = _pair_db_paths(db_paths)
        results: list[tuple[int, dict[str, float], dict[str, float]]] = []
        loaded_run_ids: set[int] = set()

        for run_id, macro_db, micro_db in pairs:
            macro_best = self.load_best_params(macro_db, f"{scenario.name}-{role}-macro-{run_id:03d}")
            micro_best = self.load_best_params(micro_db, f"{scenario.name}-{role}-micro-{run_id:03d}")
            if macro_best and micro_best:
                results.append((run_id, macro_best, micro_best))
                loaded_run_ids.add(run_id)

        for run_id, macro_db, macro_study, micro_db, micro_study in _pair_db_paths_by_study_name(
            scenario=scenario,
            role=role,
            db_paths=db_paths,
            store=self,
        ):
            if run_id in loaded_run_ids:
                continue
            macro_best = self.load_best_params(macro_db, macro_study)
            micro_best = self.load_best_params(micro_db, micro_study)
            if macro_best and micro_best:
                results.append((run_id, macro_best, micro_best))
                loaded_run_ids.add(run_id)

        return sorted(results, key=lambda item: item[0])


_DEFAULT_STORE: TrialStore = OptunaTrialStore()


def load_best_params(db_path: Path, study_name: str | None) -> dict[str, float] | None:
    """Load best trial parameters using default TrialStore.

    Args:
        db_path: Path to sqlite Optuna DB.
        study_name: Optional study name.

    Returns:
        dict[str, float] | None: Best parameter mapping when available.
    """
    return _DEFAULT_STORE.load_best_params(db_path, study_name)


def scan_runs_for_pairs(
    scenario: ScenarioConfig,
    scenario_dir: Path,
    role: Role,
) -> list[tuple[int, dict[str, float], dict[str, float]]]:
    """Scan standard run directories and return paired parameters.

    Args:
        scenario: Scenario definition.
        scenario_dir: Scenario output directory.
        role: Target role (`train` or `eval`).

    Returns:
        list[tuple[int, dict[str, float], dict[str, float]]]: Sorted paired parameter list.
    """
    run_root = runs_dir(scenario_dir, role)
    if not run_root.exists():
        return []
    db_paths = list(run_root.rglob("optuna.db"))
    return _DEFAULT_STORE.pair_trials(scenario, role, db_paths)


def scan_db_paths_for_pairs(
    scenario: ScenarioConfig,
    role: Role,
    db_paths: Sequence[Path],
) -> list[tuple[int, dict[str, float], dict[str, float]]]:
    """Pair parameters from explicit DB paths.

    Args:
        scenario: Scenario definition.
        role: Target role (`train` or `eval`).
        db_paths: Candidate DB file paths.

    Returns:
        list[tuple[int, dict[str, float], dict[str, float]]]: Sorted paired parameter list.
    """
    return _DEFAULT_STORE.pair_trials(scenario, role, db_paths)


def load_pairs_from_db_pairs(
    scenario: ScenarioConfig,
    role: Role,
    db_pairs: Sequence[tuple[Path, Path]],
    store: TrialStore | None = None,
) -> list[tuple[int, dict[str, float], dict[str, float]]]:
    """Load paired parameters from explicit macro/micro DB pairs.

    Args:
        scenario: Scenario definition.
        role: Target role (`train` or `eval`).
        db_pairs: Explicit `(macro_db, micro_db)` pairs.
        store: Optional store implementation for dependency injection.

    Returns:
        list[tuple[int, dict[str, float], dict[str, float]]]: Paired parameter list.
    """
    _ = scenario
    _ = role
    trial_store = store or _DEFAULT_STORE
    results: list[tuple[int, dict[str, float], dict[str, float]]] = []
    for idx, (macro_db, micro_db) in enumerate(db_pairs):
        macro_best = trial_store.load_best_params(macro_db, None)
        micro_best = trial_store.load_best_params(micro_db, None)
        if macro_best and micro_best:
            results.append((idx, macro_best, micro_best))
    return results


def _pair_db_paths(db_paths: Iterable[Path]) -> list[tuple[int, Path, Path]]:
    """Pair DB paths by inferred run id and target."""
    grouped: dict[int, dict[str, Path]] = {}
    for db_path in db_paths:
        target = _infer_target(db_path)
        run_id = _infer_run_id(db_path)
        if target is None or run_id is None:
            continue
        grouped.setdefault(run_id, {})[target] = db_path

    pairs: list[tuple[int, Path, Path]] = []
    for run_id, targets in grouped.items():
        macro_db = targets.get("macro")
        micro_db = targets.get("micro")
        if macro_db and micro_db:
            pairs.append((run_id, macro_db, micro_db))
    return sorted(pairs, key=lambda item: item[0])


def _infer_target(db_path: Path) -> str | None:
    """Infer target kind (`macro` or `micro`) from DB path."""
    parent = db_path.parent.name
    if parent in {"macro", "micro"}:
        return parent
    return None


def _infer_run_id(db_path: Path) -> int | None:
    """Infer integer run id from DB path."""
    run_dir_path = db_path.parent.parent
    try:
        return int(run_dir_path.name)
    except ValueError:
        return None


def _pair_db_paths_by_study_name(
    scenario: ScenarioConfig,
    role: Role,
    db_paths: Sequence[Path],
    store: TrialStore,
) -> list[tuple[int, Path, str, Path, str]]:
    """Pair DB paths by matching study name convention."""
    grouped: dict[int, dict[str, tuple[Path, str]]] = {}
    pattern = re.compile(rf"^{re.escape(scenario.name)}-{role}-(?P<target>macro|micro)-(?P<run_id>\d{{3}})$")

    for db_path in db_paths:
        for study_name in store.list_studies(db_path):
            match = pattern.fullmatch(study_name)
            if not match:
                continue
            target = match.group("target")
            run_id = int(match.group("run_id"))
            grouped.setdefault(run_id, {})[target] = (db_path, study_name)

    pairs: list[tuple[int, Path, str, Path, str]] = []
    for run_id, targets in grouped.items():
        macro = targets.get("macro")
        micro = targets.get("micro")
        if macro and micro:
            pairs.append((run_id, macro[0], macro[1], micro[0], micro[1]))
    return sorted(pairs, key=lambda item: item[0])
