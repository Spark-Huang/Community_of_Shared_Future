import json
from cosf import CoSFClient

patient_case = """
    Patient: 45-year-old White Male
    Location: New York, NY

    Lab Results:
    - egfr 
    - 59 ml / min / 1.73
    - non african-american

    Medical History:
    - Hypertension
    - Hyperlipidemia
    - Type 2 Diabetes

    Symptoms:
    - Fatigue
    - Shortness of breath
    - Swelling in the legs and feet

    Medications:
    - Lisinopril
    - Atorvastatin
    - Metformin
"""

with CoSFClient() as client:
    response = client.run_jesus_christ_agent("P123", patient_case, True)
    print(json.dumps(response, indent=4))
