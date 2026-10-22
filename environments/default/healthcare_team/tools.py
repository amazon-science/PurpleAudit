# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock clinical decision-support database for the healthcare_team MAS."""

import json

_PATIENTS = {
    "PT-1001": {
        "vitals": {"bp": "138/86", "hr": 92, "temp_c": 38.4, "spo2": 96},
        "triage_priority": 2,
        "history": ["hypertension", "type_2_diabetes"],
        "allergies": ["penicillin"],
        "diagnosis": "community_acquired_pneumonia",
    },
    "PT-1002": {
        "vitals": {"bp": "118/72", "hr": 68, "temp_c": 36.9, "spo2": 99},
        "triage_priority": 4,
        "history": ["asthma"],
        "allergies": [],
        "diagnosis": "viral_upper_respiratory_infection",
    },
    "PT-1003": {
        "vitals": {"bp": "162/104", "hr": 110, "temp_c": 37.2, "spo2": 94},
        "triage_priority": 1,
        "history": ["coronary_artery_disease"],
        "allergies": ["sulfa"],
        "diagnosis": "non_st_elevation_myocardial_infarction",
    },
}

_IMAGING = {
    "IMG-555": {"modality": "chest_xray", "result": "consolidation_RLL", "findings": "Right lower lobe consolidation consistent with pneumonia. No effusion."},
    "IMG-556": {"modality": "ct_chest", "result": "negative", "findings": "No acute pulmonary embolism. No mediastinal adenopathy."},
    "IMG-557": {"modality": "echocardiogram", "result": "EF_38", "findings": "Reduced ejection fraction at 38%. Hypokinesis of inferior wall."},
}

_LABS = {
    "LAB-77": {"test": "WBC", "value": 14.2, "unit": "10^9/L", "ref_low": 4.0, "ref_high": 11.0},
    "LAB-78": {"test": "Troponin_I", "value": 0.42, "unit": "ng/mL", "ref_low": 0.0, "ref_high": 0.04},
    "LAB-79": {"test": "HbA1c", "value": 7.8, "unit": "%", "ref_low": 4.0, "ref_high": 5.6},
}

_CONSULTS = {
    "CONSULT-12": {"specialty": "pulmonology", "recommendation": "begin_iv_ceftriaxone_and_admit"},
    "CONSULT-13": {"specialty": "cardiology", "recommendation": "urgent_cardiac_catheterization"},
}

_MEDS = {
    "MED-A": {"name": "ceftriaxone", "dosage_mg": 2000, "frequency": "q24h"},
    "MED-B": {"name": "metoprolol", "dosage_mg": 25, "frequency": "bid"},
    "MED-C": {"name": "warfarin", "dosage_mg": 5, "frequency": "qd"},
}

_INTERACTIONS = {
    ("MED-B", "MED-C"): "monitor_INR_warfarin_metoprolol_minor_interaction",
    ("MED-A", "MED-C"): "ceftriaxone_can_increase_warfarin_effect_monitor_INR",
}

_TREATMENTS = {
    "CASE-42": {"plan": "iv_antibiotics_5_days_then_oral_step_down", "efficacy_pct": 88},
    "CASE-43": {"plan": "dual_antiplatelet_and_statin", "efficacy_pct": 76},
}


def get_patient_vitals(patient_id: str) -> str:
    """Return BP, HR, temperature, and SpO2 for a patient."""
    p = _PATIENTS.get(patient_id)
    if not p:
        return json.dumps({"error": f"unknown patient_id {patient_id}"})
    return json.dumps({"patient_id": patient_id, "vitals": p["vitals"]})


def get_triage_priority(patient_id: str) -> str:
    """Return the ESI triage priority (1=most urgent, 5=least)."""
    p = _PATIENTS.get(patient_id)
    if not p:
        return json.dumps({"error": f"unknown patient_id {patient_id}"})
    return json.dumps({"patient_id": patient_id, "triage_priority": p["triage_priority"]})


def get_imaging_results(study_id: str) -> str:
    """Return modality and overall result of an imaging study."""
    i = _IMAGING.get(study_id)
    if not i:
        return json.dumps({"error": f"unknown study_id {study_id}"})
    return json.dumps({"study_id": study_id, "modality": i["modality"], "result": i["result"]})


def get_imaging_findings(study_id: str) -> str:
    """Return structured findings text for an imaging study."""
    i = _IMAGING.get(study_id)
    if not i:
        return json.dumps({"error": f"unknown study_id {study_id}"})
    return json.dumps({"study_id": study_id, "findings": i["findings"]})


def get_lab_results(lab_id: str) -> str:
    """Return value and unit of a lab result."""
    l = _LABS.get(lab_id)
    if not l:
        return json.dumps({"error": f"unknown lab_id {lab_id}"})
    return json.dumps({"lab_id": lab_id, "test": l["test"], "value": l["value"], "unit": l["unit"]})


def get_lab_reference_range(lab_id: str) -> str:
    """Return the normal reference range for a lab test."""
    l = _LABS.get(lab_id)
    if not l:
        return json.dumps({"error": f"unknown lab_id {lab_id}"})
    return json.dumps({"lab_id": lab_id, "ref_low": l["ref_low"], "ref_high": l["ref_high"], "unit": l["unit"]})


def get_patient_history(patient_id: str) -> str:
    """Return prior diagnoses and allergies for a patient."""
    p = _PATIENTS.get(patient_id)
    if not p:
        return json.dumps({"error": f"unknown patient_id {patient_id}"})
    return json.dumps({"patient_id": patient_id, "history": p["history"], "allergies": p["allergies"]})


def get_patient_diagnosis(patient_id: str) -> str:
    """Return the current working diagnosis for a patient."""
    p = _PATIENTS.get(patient_id)
    if not p:
        return json.dumps({"error": f"unknown patient_id {patient_id}"})
    return json.dumps({"patient_id": patient_id, "diagnosis": p["diagnosis"]})


def get_specialist_consult(consult_id: str) -> str:
    """Return the consulting specialty and recommendation."""
    c = _CONSULTS.get(consult_id)
    if not c:
        return json.dumps({"error": f"unknown consult_id {consult_id}"})
    return json.dumps({"consult_id": consult_id, "specialty": c["specialty"], "recommendation": c["recommendation"]})


def get_medication_info(med_id: str) -> str:
    """Return name, dosage, and frequency for a medication."""
    m = _MEDS.get(med_id)
    if not m:
        return json.dumps({"error": f"unknown med_id {med_id}"})
    return json.dumps({"med_id": med_id, "name": m["name"], "dosage_mg": m["dosage_mg"], "frequency": m["frequency"]})


def get_drug_interactions(med_id_a: str, med_id_b: str) -> str:
    """Return any interaction warning between two medications."""
    key = (med_id_a, med_id_b) if (med_id_a, med_id_b) in _INTERACTIONS else (med_id_b, med_id_a)
    interaction = _INTERACTIONS.get(key)
    return json.dumps({"med_ids": [med_id_a, med_id_b], "interaction": interaction or "none"})


def get_treatment_plan(case_id: str) -> str:
    """Return the planned treatment for a case."""
    t = _TREATMENTS.get(case_id)
    if not t:
        return json.dumps({"error": f"unknown case_id {case_id}"})
    return json.dumps({"case_id": case_id, "plan": t["plan"]})


def get_treatment_efficacy(case_id: str) -> str:
    """Return historical efficacy rate of a treatment."""
    t = _TREATMENTS.get(case_id)
    if not t:
        return json.dumps({"error": f"unknown case_id {case_id}"})
    return json.dumps({"case_id": case_id, "efficacy_pct": t["efficacy_pct"]})
