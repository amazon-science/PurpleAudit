# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock case-law and document database for the legal_research_team MAS."""

import json

_CASES = {
    "CASE-2019-432": {
        "summary": "Plaintiff sought injunction against trade-secret misappropriation by former employee.",
        "outcome": "won",
        "jurisdiction": "9th_Circuit",
        "topic": "trade_secret",
        "citation": "432 F.3d 921 (9th Cir. 2019)",
    },
    "CASE-2017-118": {
        "summary": "Class-action securities-fraud claim under Section 10(b) survived motion to dismiss.",
        "outcome": "settled",
        "jurisdiction": "SDNY",
        "topic": "securities",
        "citation": "118 F. Supp. 3d 217 (S.D.N.Y. 2017)",
    },
    "CASE-2021-885": {
        "summary": "Patent-infringement suit on graph-database indexing technology.",
        "outcome": "lost",
        "jurisdiction": "Federal_Circuit",
        "topic": "patent",
        "citation": "885 F.3d 1042 (Fed. Cir. 2021)",
    },
    "CASE-2023-204": {
        "summary": "FTC challenge to vertical merger of two enterprise SaaS providers.",
        "outcome": "won",
        "jurisdiction": "DC_Circuit",
        "topic": "antitrust",
        "citation": "204 F.4th 88 (D.C. Cir. 2023)",
    },
}

_STATUTES = {
    "USC-15-1": {"text": "Sherman Antitrust Act, Section 1: agreements in restraint of trade are illegal.", "jurisdiction": "Federal"},
    "USC-17-501": {"text": "Copyright Act, Section 501: defines copyright infringement.", "jurisdiction": "Federal"},
    "CAL-CC-3426": {"text": "California Uniform Trade Secrets Act, Section 3426: civil remedies for misappropriation.", "jurisdiction": "California"},
}

_TEMPLATES = {
    "NDA": {"content": "MUTUAL NON-DISCLOSURE AGREEMENT — definitions, term, exclusions, remedies.", "status": "current"},
    "MSA": {"content": "MASTER SERVICES AGREEMENT — scope, payment, IP, indemnification.", "status": "current"},
    "MOU": {"content": "MEMORANDUM OF UNDERSTANDING — non-binding statement of intent.", "status": "deprecated"},
}


def search_cases(query: str) -> str:
    """Return case IDs whose topic matches the query."""
    q = query.lower()
    matches = [cid for cid, c in _CASES.items() if q in c["topic"]]
    return json.dumps({"query": query, "case_ids": matches})


def get_case_summary(case_id: str) -> str:
    """Return a one-paragraph summary of a case."""
    c = _CASES.get(case_id)
    if not c:
        return json.dumps({"error": f"unknown case_id {case_id}"})
    return json.dumps({"case_id": case_id, "summary": c["summary"]})


def get_case_outcome(case_id: str) -> str:
    """Return the outcome of a case (won/lost/settled)."""
    c = _CASES.get(case_id)
    if not c:
        return json.dumps({"error": f"unknown case_id {case_id}"})
    return json.dumps({"case_id": case_id, "outcome": c["outcome"]})


def get_case_jurisdiction(case_id: str) -> str:
    """Return the jurisdiction of a case."""
    c = _CASES.get(case_id)
    if not c:
        return json.dumps({"error": f"unknown case_id {case_id}"})
    return json.dumps({"case_id": case_id, "jurisdiction": c["jurisdiction"]})


def format_citation(case_id: str) -> str:
    """Return the Bluebook citation for a case."""
    c = _CASES.get(case_id)
    if not c:
        return json.dumps({"error": f"unknown case_id {case_id}"})
    return json.dumps({"case_id": case_id, "citation": c["citation"]})


def get_statute_text(code: str) -> str:
    """Return the text of a statute by its code."""
    s = _STATUTES.get(code)
    if not s:
        return json.dumps({"error": f"unknown statute code {code}"})
    return json.dumps({"code": code, "text": s["text"]})


def get_statute_jurisdiction(code: str) -> str:
    """Return the jurisdiction of a statute."""
    s = _STATUTES.get(code)
    if not s:
        return json.dumps({"error": f"unknown statute code {code}"})
    return json.dumps({"code": code, "jurisdiction": s["jurisdiction"]})


def get_document_template(doc_type: str) -> str:
    """Return the template content for a document type."""
    t = _TEMPLATES.get(doc_type)
    if not t:
        return json.dumps({"error": f"unknown template {doc_type}"})
    return json.dumps({"doc_type": doc_type, "content": t["content"]})


def get_template_status(doc_type: str) -> str:
    """Return whether a template is current or deprecated."""
    t = _TEMPLATES.get(doc_type)
    if not t:
        return json.dumps({"error": f"unknown template {doc_type}"})
    return json.dumps({"doc_type": doc_type, "status": t["status"]})
