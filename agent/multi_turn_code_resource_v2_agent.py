from .multi_turn_code_resource_agent import MultiTurnCodeResourceAgent


# Extra guidance on top of the FHIR-AgentBench baseline prompt. Addresses failure modes seen in the
# baseline: guessing field names/values instead of inspecting the data, exact-match filtering, and
# never fetching Medication resources (MedicationRequest only references them).
DATA_GUIDANCE = """

Additional guidance on this FHIR data (MIMIC-IV on FHIR):
- Only the `answer` variable is returned to you from execute_python_code; print() output is NOT shown to you.
  Never call exit(), sys.exit() or raise SystemExit.
- Inspect before you filter. Questions use natural-language wording that often differs from the stored values.
  Before filtering by a name, run code that sets `answer` to the distinct candidate values (e.g. all code displays
  containing a keyword, or one example resource), then choose the matching ones in a follow-up call.
  Never conclude "not found" or 0 without having inspected the candidates you searched.
- Field shapes: Observation.category and Encounter.identifier are lists; category names are in category[].coding[].code.
- Match names case-insensitively and partially (substring in either direction, ignoring punctuation), not by exact equality.
- Drug names are NOT stored on MedicationRequest. A MedicationRequest has medicationReference.reference = "Medication/<id>".
  Retrieve Medication resources as well (get_resources_by_patient_fhir_id with resource_type='Medication' returns the
  medication catalog) and read the name from Medication.identifier where system ends with 'mimic-medication-name'.
  Prescription time is MedicationRequest.authoredOn.
- Observation covers labs, vital signs, and charted input/output events. Use Observation.category codes
  (e.g. 'Output') and Observation.code.coding[].display to find the right item; values are in valueQuantity.value.
- Procedures are in Procedure.code.coding[].display with time in performedDateTime; diagnoses are in Condition.code.coding[].display.

Example: finding prescriptions of a drug by name
```python
meds = {m['id']: m for m in retrieved_resources['Medication']}
def med_name(m):
    return next((i['value'] for i in m.get('identifier', []) if i.get('system', '').endswith('mimic-medication-name')), '')
names = {mid: med_name(m).lower() for mid, m in meds.items()}
answer = sorted({n for n in names.values() if 'insulin' in n})  # call 1: inspect candidate names
```
```python
# call 2: after reviewing the candidates
reqs = [r for r in retrieved_resources['MedicationRequest']
        if 'insulin' in names.get(r.get('medicationReference', {}).get('reference', '').split('/')[-1], '')]
answer = [(r['id'], r.get('authoredOn')) for r in reqs]
```"""


class MultiTurnCodeResourceV2Agent(MultiTurnCodeResourceAgent):
    """Multi-turn code agent with extra guidance on the MIMIC-IV FHIR data layout and on inspecting values before filtering."""

    def __init__(self, model: str, max_iterations: int = 30, verbose: bool = False, base_url=None):
        super().__init__(model, max_iterations=max_iterations, verbose=verbose, base_url=base_url)
        self.system_msg = [{"role": "system", "content": self.system_msg[0]["content"] + DATA_GUIDANCE}]
