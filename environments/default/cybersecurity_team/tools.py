# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock SOC database for the cybersecurity_team MAS."""

import json

_THREATS = {
    "THR-2024-001": {"type": "phishing", "severity": "high", "source_ip": "203.0.113.42", "target": "email_gateway"},
    "THR-2024-002": {"type": "DDoS", "severity": "medium", "source_ip": "198.51.100.7", "target": "web_server"},
    "THR-2024-003": {"type": "malware", "severity": "critical", "source_ip": "192.0.2.55", "target": "endpoint_workstation"},
    "THR-2024-004": {"type": "ransomware", "severity": "critical", "source_ip": "203.0.113.99", "target": "file_server"},
}

_VULNERABILITIES = {
    "CVE-2021-44228": {"severity": "critical", "cvss": 10.0, "status": "patched", "name": "Log4Shell"},
    "CVE-2024-1234": {"severity": "high", "cvss": 8.1, "status": "in_progress", "name": "OpenSSL_RCE"},
    "CVE-2023-7891": {"severity": "medium", "cvss": 5.4, "status": "patched", "name": "NginxRequestSmuggling"},
    "CVE-2024-9012": {"severity": "low", "cvss": 3.7, "status": "wontfix", "name": "InfoLeak_Telemetry"},
}

_INCIDENTS = {
    "INC-2024-001": {"status": "investigating", "assignee": "David_Kim", "priority": "P1"},
    "INC-2024-002": {"status": "resolved", "assignee": "Sara_Chen", "priority": "P3"},
    "INC-2024-003": {"status": "open", "assignee": "Marcus_Lee", "priority": "P2"},
    "INC-2024-004": {"status": "investigating", "assignee": "Priya_Nair", "priority": "P1"},
}

_PATCH_STATUS = {
    "web_server": {"status": "current", "last_patch": "2024-09-12"},
    "email_gateway": {"status": "behind", "last_patch": "2024-06-20"},
    "endpoint_workstation": {"status": "current", "last_patch": "2024-09-15"},
    "file_server": {"status": "behind", "last_patch": "2024-05-03"},
}


def get_threat_type(threat_id: str) -> str:
    """Return the type of a threat (e.g. malware, phishing, DDoS)."""
    rec = _THREATS.get(threat_id)
    if not rec:
        return json.dumps({"error": f"unknown threat_id {threat_id}"})
    return json.dumps({"threat_id": threat_id, "type": rec["type"]})


def get_threat_severity(threat_id: str) -> str:
    """Return the severity level of a threat."""
    rec = _THREATS.get(threat_id)
    if not rec:
        return json.dumps({"error": f"unknown threat_id {threat_id}"})
    return json.dumps({"threat_id": threat_id, "severity": rec["severity"]})


def get_threat_source_ip(threat_id: str) -> str:
    """Return the source IP address of a threat."""
    rec = _THREATS.get(threat_id)
    if not rec:
        return json.dumps({"error": f"unknown threat_id {threat_id}"})
    return json.dumps({"threat_id": threat_id, "source_ip": rec["source_ip"]})


def get_threat_target(threat_id: str) -> str:
    """Return the target system of a threat."""
    rec = _THREATS.get(threat_id)
    if not rec:
        return json.dumps({"error": f"unknown threat_id {threat_id}"})
    return json.dumps({"threat_id": threat_id, "target": rec["target"]})


def get_active_threats() -> str:
    """List all currently active threats."""
    return json.dumps({"active_threats": list(_THREATS.keys())})


def get_vulnerability_severity(cve: str) -> str:
    """Return the severity of a CVE."""
    rec = _VULNERABILITIES.get(cve)
    if not rec:
        return json.dumps({"error": f"unknown cve {cve}"})
    return json.dumps({"cve": cve, "severity": rec["severity"]})


def get_vulnerability_cvss(cve: str) -> str:
    """Return the CVSS score of a CVE."""
    rec = _VULNERABILITIES.get(cve)
    if not rec:
        return json.dumps({"error": f"unknown cve {cve}"})
    return json.dumps({"cve": cve, "cvss": rec["cvss"]})


def get_vulnerability_status(cve: str) -> str:
    """Return the patch/remediation status of a CVE."""
    rec = _VULNERABILITIES.get(cve)
    if not rec:
        return json.dumps({"error": f"unknown cve {cve}"})
    return json.dumps({"cve": cve, "status": rec["status"]})


def get_vulnerabilities_by_criticality(severity: str) -> str:
    """List CVEs at the given criticality level."""
    matches = [c for c, r in _VULNERABILITIES.items() if r["severity"] == severity]
    return json.dumps({"severity": severity, "cves": matches})


def get_patch_status(system: str) -> str:
    """Return patch deployment status for a system."""
    rec = _PATCH_STATUS.get(system)
    if not rec:
        return json.dumps({"error": f"unknown system {system}"})
    return json.dumps({"system": system, "status": rec["status"], "last_patch": rec["last_patch"]})


def get_incident_status(incident_id: str) -> str:
    """Return the status of an incident."""
    rec = _INCIDENTS.get(incident_id)
    if not rec:
        return json.dumps({"error": f"unknown incident_id {incident_id}"})
    return json.dumps({"incident_id": incident_id, "status": rec["status"]})


def get_incident_assignee(incident_id: str) -> str:
    """Return the responder assigned to an incident."""
    rec = _INCIDENTS.get(incident_id)
    if not rec:
        return json.dumps({"error": f"unknown incident_id {incident_id}"})
    return json.dumps({"incident_id": incident_id, "assignee": rec["assignee"]})


def get_incident_priority(incident_id: str) -> str:
    """Return the priority of an incident."""
    rec = _INCIDENTS.get(incident_id)
    if not rec:
        return json.dumps({"error": f"unknown incident_id {incident_id}"})
    return json.dumps({"incident_id": incident_id, "priority": rec["priority"]})


def get_active_incidents() -> str:
    """List currently active (non-resolved) incidents."""
    active = [i for i, r in _INCIDENTS.items() if r["status"] != "resolved"]
    return json.dumps({"active_incidents": active})
