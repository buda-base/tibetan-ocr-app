"""
Artifact Management for OCR Pipeline.

Handles structured storage of all artifacts generated during OCR processing.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import numpy.typing as npt

SUBDIR_NAMES = ["detection", "dewarping", "lines", "results"]


class ArtifactManager:
    """Manages structured artifact storage for OCR pipeline outputs."""

    def __init__(self, base_output_dir: str, job_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.base_output_dir = Path(base_output_dir)
        self.job_id = job_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}"
        self.job_dir = self.base_output_dir / self.job_id
        self.config = config or {}
        self.manifest: List[Dict[str, str]] = []
        self.page_metrics: Dict[str, Dict[str, Any]] = {}
        self.current_page: Optional[str] = None
        self._base_dir: Path = self.job_dir  # Where subdirs are rooted

    @property
    def subdirs(self) -> Dict[str, Path]:
        return {name: self._base_dir / name for name in SUBDIR_NAMES}

    def create_directory_structure(self):
        """Create the base job directory."""
        self.job_dir.mkdir(parents=True, exist_ok=True)

    def set_current_page(self, page_name: str):
        """Set current page context for batch processing (creates page subdirectory)."""
        self.current_page = page_name
        self._base_dir = self.job_dir / page_name
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_subdir(self, subdir: str) -> Path:
        """Ensure subdirectory exists and return its path."""
        if subdir not in SUBDIR_NAMES:
            raise ValueError(f"Unknown subdirectory: {subdir}")
        path = self.subdirs[subdir]
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_results_dir(self) -> Path:
        """Get results directory path, creating it if necessary."""
        return self._ensure_subdir("results")

    def _add_to_manifest(self, name: str, artifact_type: str, path: str):
        self.manifest.append(
            {"name": name, "type": artifact_type, "path": path, "timestamp": datetime.now().isoformat()}
        )

    def save_config(self):
        """Save job configuration to config.json."""
        path = self.job_dir / "config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, default=str)
        self._add_to_manifest("config.json", "configuration", str(path))

    def save_image(self, name: str, image: npt.NDArray, subdir: str, fmt: str = "png") -> Path:
        """Save an image artifact."""
        path = self._ensure_subdir(subdir) / f"{name}.{fmt}"
        cv2.imwrite(str(path), image)
        self._add_to_manifest(name, "image", str(path))
        return path

    def save_json(self, name: str, data: Any, subdir: str) -> Path:
        """Save JSON data artifact."""
        path = self._ensure_subdir(subdir) / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        self._add_to_manifest(name, "json", str(path))
        return path

    def save_numpy(self, name: str, array: npt.NDArray, subdir: str) -> Path:
        """Save numpy array artifact."""
        path = self._ensure_subdir(subdir) / f"{name}.npy"
        np.save(path, array)
        self._add_to_manifest(name, "numpy", str(path))
        return path

    def save_text(self, name: str, text: str, subdir: str, ext: str = "txt") -> Path:
        """Save text artifact."""
        path = self._ensure_subdir(subdir) / f"{name}.{ext}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        self._add_to_manifest(name, "text", str(path))
        return path

    def generate_manifest(self) -> Path:
        """Generate and save the artifact manifest."""
        path = self.job_dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"job_id": self.job_id, "created": datetime.now().isoformat(), "artifacts": self.manifest}, f, indent=2
            )
        return path

    def save_metrics(self, metrics: Dict[str, Any]) -> Path:
        """Save metrics (stores for aggregation in batch mode, writes directly otherwise)."""
        if self.current_page:
            self.page_metrics[self.current_page] = metrics
            return self.job_dir / "metrics.json"
        path = self.job_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)
        return path

    def save_aggregate_metrics(self) -> Path:
        """Save aggregated metrics for batch job."""
        total_duration = sum(m.get("total_duration_ms", 0) for m in self.page_metrics.values())
        total_lines = sum(m.get("lines_detected", 0) for m in self.page_metrics.values())
        successful = sum(1 for m in self.page_metrics.values() if m.get("status") != "failed")

        aggregate = {
            "job_summary": {
                "total_pages": len(self.page_metrics),
                "successful_pages": successful,
                "total_duration_ms": total_duration,
                "avg_duration_per_page_ms": total_duration / len(self.page_metrics) if self.page_metrics else 0,
                "total_lines_detected": total_lines,
            },
            "per_page_metrics": self.page_metrics,
        }
        path = self.job_dir / "metrics.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2, default=str)
        return path
