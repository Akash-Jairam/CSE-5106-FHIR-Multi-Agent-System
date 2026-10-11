# Load the MIMIC-IV Clinical Database Demo on FHIR (v2.1.0) into a GCP FHIR store.
# Requires: gcloud (authenticated), curl.
# Usage:
#   PROJECT_ID=... LOCATION=us-central1 DATASET_ID=mimic-fhir STORE_ID=mimic-fhir \
#   BUCKET=my-mimic-fhir-bucket bash scripts/setup_fhir_store.sh
set -euo pipefail

for var in PROJECT_ID LOCATION DATASET_ID STORE_ID BUCKET; do
  if [ -z "${!var:-}" ]; then
    echo "❌ Missing required environment variable: $var"
    exit 1
  fi
done

FHIR_DEMO_URL="https://physionet.org/files/mimic-iv-fhir-demo/2.1.0/"
LOCAL_DIR="physionet.org/files/mimic-iv-fhir-demo/2.1.0"
GCS_PREFIX="gs://${BUCKET}/mimic-fhir"

echo "🔧 Enabling Cloud Healthcare API..."
gcloud services enable healthcare.googleapis.com --project="$PROJECT_ID"

echo "📥 Downloading MIMIC-IV FHIR demo v2.1.0..."
mkdir -p "$LOCAL_DIR/fhir"
for f in $(curl -fsSL "${FHIR_DEMO_URL}fhir/" | grep -o 'href="[^"]*\.ndjson\.gz"' | sed 's/href="//; s/"$//'); do
  if [ ! -f "$LOCAL_DIR/fhir/${f%.gz}" ]; then
    echo "   $f"
    # Retry and resume (-C -) since large files can time out on PhysioNet
    curl -fsSL --retry 10 --retry-all-errors --retry-delay 5 -C - \
      -o "$LOCAL_DIR/fhir/$f" "${FHIR_DEMO_URL}fhir/$f"
    gunzip -f "$LOCAL_DIR/fhir/$f"
  fi
done

echo "🪣 Uploading NDJSON files to ${GCS_PREFIX}..."
if ! gcloud storage buckets describe "gs://${BUCKET}" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" --project="$PROJECT_ID" --location="$LOCATION"
fi
# --no-clobber skips files already uploaded by a previous run
gcloud storage cp --no-clobber "$LOCAL_DIR"/fhir/*.ndjson "${GCS_PREFIX}/"

echo "🔐 Granting the Healthcare service agent read access to the bucket..."
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-healthcare.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer" >/dev/null

echo "🏥 Creating dataset and FHIR store (skipped if they exist)..."
if ! gcloud healthcare datasets describe "$DATASET_ID" --location="$LOCATION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud healthcare datasets create "$DATASET_ID" --location="$LOCATION" --project="$PROJECT_ID"
fi
if ! gcloud healthcare fhir-stores describe "$STORE_ID" --dataset="$DATASET_ID" --location="$LOCATION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  gcloud healthcare fhir-stores create "$STORE_ID" \
    --dataset="$DATASET_ID" --location="$LOCATION" --project="$PROJECT_ID" \
    --version=R4 --enable-update-create
fi

echo "📤 Importing resources (this usually takes ~10 minutes)..."
gcloud healthcare fhir-stores import gcs "$STORE_ID" \
  --dataset="$DATASET_ID" --location="$LOCATION" --project="$PROJECT_ID" \
  --gcs-uri="${GCS_PREFIX}/*.ndjson" \
  --content-structure=resource

echo "✅ Import complete. Add these to config.yml under FHIR_CONFIG:"
echo "   PROJECT_ID: \"$PROJECT_ID\""
echo "   LOCATION: \"$LOCATION\""
echo "   DATASET_ID: \"$DATASET_ID\""
echo "   STORE_ID: \"$STORE_ID\""
echo "Then run: python scripts/check_setup.py"
