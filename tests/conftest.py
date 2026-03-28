"""Shared test fixtures.

Pre-seeds src.config with fake credentials so that importing the module
does not require real GCP ADC.  Tests that need real credentials (integration
tests) call config.setup_clients() themselves, which re-resolves as needed.
"""

from unittest.mock import MagicMock

import src.config as config

# Pre-seed the lazy credential cache so no test triggers google.auth.default()
config._credentials = MagicMock(name="FakeCredentials")
config._project_id = "test-project"
config._resolved = True
