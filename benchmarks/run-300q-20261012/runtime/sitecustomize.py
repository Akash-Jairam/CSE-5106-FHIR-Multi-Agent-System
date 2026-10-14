"""Bootstrap the adapter in launcher subprocesses and spawned workers only."""
import os
import sys

config = os.getenv("FHIR_LOCAL_AZURE_CONFIG")
if config:
    try:
        from azure_adapter import AzureSettings, install
        install(AzureSettings.load(config))
        from pathlib import Path
        import utils.core_utils as core
        core.CONFIG_PATH = Path(config)
    except Exception:
        # Avoid falling through to the normal OpenAI route after a bootstrap
        # failure. Do not display configuration values or secret-bearing errors.
        print("Local Azure setup could not load. Run fhir-azure info.",
              file=sys.stderr, flush=True)
        os._exit(2)
