"""Verify that config.yml, GCP credentials, and the MIMIC-IV FHIR store are ready.

Usage: python scripts/check_setup.py [--model gpt-6-luna]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import get_default_base_url, get_default_model, validate_config

# Patient used by check_tool_credentials() and the first questions in the dataset
SMOKE_PATIENT_FHIR_ID = "dd2bf984-33c3-5874-8f68-84113327877e"
EXPECTED_PATIENT_COUNT = 100  # MIMIC-IV FHIR demo v2.1.0


def count_resources(client, resource_type):
    # Cloud Healthcare API doesn't support _summary=count; a 1-entry search still returns "total"
    response = client.session.get(f"{client.fhir_store_url}/{resource_type}?_count=1")
    response.raise_for_status()
    return response.json().get("total", 0)


def main():
    parser = argparse.ArgumentParser(description="Check environment and FHIR store setup.")
    parser.add_argument("--model", type=str, default=get_default_model(),
                        help="Model whose API key should be checked; default: the configured base model")
    args = parser.parse_args()

    print("1. Config")
    if not validate_config(args.model, get_default_base_url(args.model)):
        print("   Fill in config.yml (see config.example.yml) or set the matching environment variables.")
        sys.exit(1)

    print("2. Google Application Default Credentials")
    try:
        from fhir_client import get_fhir_client
        client = get_fhir_client()
    except Exception as e:
        print(f"❌ Could not load credentials: {e}")
        print("   Run: gcloud auth application-default login")
        sys.exit(1)
    print("✅ Credentials loaded")

    print("3. FHIR store reachable")
    try:
        n_patients = count_resources(client, "Patient")
    except Exception as e:
        print(f"❌ Could not query {client.fhir_store_url}: {e}")
        sys.exit(1)
    if n_patients != EXPECTED_PATIENT_COUNT:
        print(f"FHIR store has {n_patients} Patient resources; expected {EXPECTED_PATIENT_COUNT}. "
              "Check that the demo import completed into the intended store.")
        sys.exit(1)
    print(f"✅ {n_patients} Patient resources (expected {EXPECTED_PATIENT_COUNT})")

    print("4. Smoke patient")
    try:
        patients = client.get_resources_by_resource_ids("Patient", [SMOKE_PATIENT_FHIR_ID])
    except Exception as e:
        print(f"❌ Lookup failed: {e}")
        sys.exit(1)
    if not patients:
        print(f"❌ Patient/{SMOKE_PATIENT_FHIR_ID} not found")
        sys.exit(1)
    print(f"✅ Patient/{SMOKE_PATIENT_FHIR_ID} found")

    print("5. Clinical resources")
    ok = True
    for resource_type in ["Encounter", "Observation", "MedicationRequest"]:
        try:
            n = count_resources(client, resource_type)
        except Exception as e:
            print(f"❌ {resource_type}: {e}")
            ok = False
            continue
        print(f"{'✅' if n > 0 else '❌'} {n} {resource_type} resources")
        ok = ok and n > 0
    if not ok:
        sys.exit(1)

    print("\n🎉 Setup looks good.")


if __name__ == "__main__":
    main()
