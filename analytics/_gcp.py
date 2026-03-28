"""Shared GCP BigQuery client factory for analytics scripts.

Resolves the project ID from (in order):
  1. --project CLI flag
  2. GOOGLE_CLOUD_PROJECT env var
  3. google.auth.default() credentials
  4. quota_project_id from ADC JSON

Usage in scripts:
    from _gcp import get_bq_client
    bq = get_bq_client()
"""

import json
import os
import sys

from google.auth import default
from google.cloud import bigquery


def _resolve_project() -> tuple:
    """Return (credentials, project_id)."""
    credentials, project = default()
    if project:
        return credentials, project

    # Fallback: read quota_project_id from ADC file
    adc_path = os.path.expanduser(
        "~/.config/gcloud/application_default_credentials.json"
    )
    if os.path.exists(adc_path):
        with open(adc_path) as f:
            adc = json.load(f)
        quota = adc.get("quota_project_id")
        if quota:
            return credentials, quota

    raise RuntimeError(
        "No GCP project found. Set GOOGLE_CLOUD_PROJECT or run:\n"
        "  gcloud config set project YOUR_PROJECT_ID"
    )


def get_bq_client(project_override: str | None = None) -> bigquery.Client:
    """Return an authenticated BigQuery client."""
    credentials, auto_project = _resolve_project()
    project = project_override or os.environ.get("GOOGLE_CLOUD_PROJECT") or auto_project
    return bigquery.Client(project=project, credentials=credentials)


def get_project_id(project_override: str | None = None) -> str:
    """Return the resolved project ID."""
    _, auto_project = _resolve_project()
    return project_override or os.environ.get("GOOGLE_CLOUD_PROJECT") or auto_project
