"""Abstract base class for all data source adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Each adapter fetches targets and their associated compound activities.

    Both methods return plain dicts matching the JSONLines schema defined in
    the design spec. The ingest orchestrator merges results from all enabled
    sources into targets.jsonl and compounds.jsonl.
    """

    @abstractmethod
    def fetch_targets(self, families: list[str]) -> list[dict]:
        """Return a list of target records.

        Each record must contain at minimum:
          id, source, gene, protein_name, organism, function,
          go_terms, diseases, target_family
        """
        ...

    @abstractmethod
    def fetch_activities(
        self,
        target_id: str,
        activity_types: list[str],
        max_compounds: int,
    ) -> list[dict]:
        """Return compound-activity records for a single target.

        Each record must contain at minimum:
          id, source, name, smiles, target_id, target_name,
          activity_type, activity_value, activity_units,
          assay_description, target_family
        """
        ...
