# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock applicant database for the hr_recruitment MAS."""

import json

_CANDIDATES = {
    "CAND-101": {"name": "Aisha_Rahman", "skills": ["python", "django", "postgres"], "experience_years": 6},
    "CAND-102": {"name": "Tomás_García", "skills": ["go", "kubernetes", "grpc"], "experience_years": 9},
    "CAND-103": {"name": "Yuna_Park", "skills": ["react", "typescript", "graphql"], "experience_years": 4},
    "CAND-104": {"name": "Daniel_Smith", "skills": ["data_engineering", "spark", "airflow"], "experience_years": 11},
}

_ROLES = {
    "ROLE-BACKEND-SR": {"title": "Senior Backend Engineer", "min_years": 5, "required_skills": ["python", "postgres"]},
    "ROLE-PLATFORM-ST": {"title": "Staff Platform Engineer", "min_years": 8, "required_skills": ["kubernetes"]},
    "ROLE-FRONTEND-MID": {"title": "Mid-level Frontend Engineer", "min_years": 3, "required_skills": ["react"]},
}

_INTERVIEWS = {
    "CAND-101": {"score": 87, "feedback": "Strong systems design, communicated trade-offs clearly.", "date": "2024-09-08"},
    "CAND-102": {"score": 92, "feedback": "Excellent distributed-systems depth, hands-on with operators.", "date": "2024-09-10"},
    "CAND-103": {"score": 74, "feedback": "Solid coding, less experience with large-scale state management.", "date": "2024-09-11"},
    "CAND-104": {"score": 81, "feedback": "Deep data-pipeline experience, lighter on streaming.", "date": "2024-09-12"},
}

_REFERENCES = {
    "CAND-101": {"status": "verified", "rating": 5},
    "CAND-102": {"status": "verified", "rating": 4},
    "CAND-103": {"status": "pending", "rating": 0},
    "CAND-104": {"status": "verified", "rating": 5},
}

_RECOMMENDATIONS = {
    "CAND-101": "hire",
    "CAND-102": "hire",
    "CAND-103": "on_hold",
    "CAND-104": "hire",
}


def get_candidate_profile(candidate_id: str) -> str:
    """Return basic profile for a candidate."""
    c = _CANDIDATES.get(candidate_id)
    if not c:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "name": c["name"]})


def get_candidate_skills(candidate_id: str) -> str:
    """Return the skill list for a candidate."""
    c = _CANDIDATES.get(candidate_id)
    if not c:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "skills": c["skills"]})


def get_candidate_experience(candidate_id: str) -> str:
    """Return the years of experience for a candidate."""
    c = _CANDIDATES.get(candidate_id)
    if not c:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "experience_years": c["experience_years"]})


def get_role_requirements(role_id: str) -> str:
    """Return the requirements (title, min years, required skills) for a role."""
    r = _ROLES.get(role_id)
    if not r:
        return json.dumps({"error": f"unknown role {role_id}"})
    return json.dumps({"role_id": role_id, "title": r["title"], "min_years": r["min_years"], "required_skills": r["required_skills"]})


def get_interview_score(candidate_id: str) -> str:
    """Return the interview score (0-100) for a candidate."""
    i = _INTERVIEWS.get(candidate_id)
    if not i:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "score": i["score"]})


def get_interview_feedback(candidate_id: str) -> str:
    """Return the textual interview feedback for a candidate."""
    i = _INTERVIEWS.get(candidate_id)
    if not i:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "feedback": i["feedback"]})


def get_interview_date(candidate_id: str) -> str:
    """Return the date of a candidate's interview."""
    i = _INTERVIEWS.get(candidate_id)
    if not i:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "date": i["date"]})


def get_reference_status(candidate_id: str) -> str:
    """Return the reference verification status for a candidate."""
    r = _REFERENCES.get(candidate_id)
    if not r:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "status": r["status"]})


def get_reference_rating(candidate_id: str) -> str:
    """Return the rating (1-5) given by the referee."""
    r = _REFERENCES.get(candidate_id)
    if not r:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "rating": r["rating"]})


def get_hiring_recommendation(candidate_id: str) -> str:
    """Return the final hiring recommendation."""
    rec = _RECOMMENDATIONS.get(candidate_id)
    if not rec:
        return json.dumps({"error": f"unknown candidate {candidate_id}"})
    return json.dumps({"candidate_id": candidate_id, "recommendation": rec})
