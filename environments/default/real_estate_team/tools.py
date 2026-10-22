# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock listings and clients database for the real_estate_team MAS."""

import json

_PROPERTIES = {
    "PROP-A100": {"address": "12_Oak_Lane", "type": "single_family", "beds": 4, "baths": 3, "sqft": 2400, "neighborhood": "Maplewood", "price": 875000, "market_value": 890000, "status": "available"},
    "PROP-A101": {"address": "55_Birch_St", "type": "condo", "beds": 2, "baths": 2, "sqft": 1100, "neighborhood": "Downtown", "price": 540000, "market_value": 535000, "status": "pending"},
    "PROP-A102": {"address": "9_Cedar_Ct", "type": "townhouse", "beds": 3, "baths": 2, "sqft": 1750, "neighborhood": "Riverside", "price": 695000, "market_value": 705000, "status": "available"},
    "PROP-A103": {"address": "201_Pine_Dr", "type": "single_family", "beds": 5, "baths": 4, "sqft": 3200, "neighborhood": "Hillside", "price": 1250000, "market_value": 1180000, "status": "sold"},
}

_NEIGHBORHOOD_STATS = {
    "Maplewood": {"avg_price_per_sqft": 365, "comps": [820000, 905000, 880000]},
    "Downtown": {"avg_price_per_sqft": 490, "comps": [515000, 560000, 545000]},
    "Riverside": {"avg_price_per_sqft": 397, "comps": [675000, 710000, 690000]},
    "Hillside": {"avg_price_per_sqft": 369, "comps": [1180000, 1210000, 1240000]},
}

_CLIENTS = {
    "CLI-301": {"budget": 900000, "neighborhood": "Maplewood", "min_beds": 4, "preferred_type": "single_family"},
    "CLI-302": {"budget": 600000, "neighborhood": "Downtown", "min_beds": 2, "preferred_type": "condo"},
    "CLI-303": {"budget": 750000, "neighborhood": "Riverside", "min_beds": 3, "preferred_type": "townhouse"},
}


def get_property_details(property_id: str) -> str:
    """Return address, type, beds, baths for a property."""
    p = _PROPERTIES.get(property_id)
    if not p:
        return json.dumps({"error": f"unknown property_id {property_id}"})
    return json.dumps({"property_id": property_id, "address": p["address"], "type": p["type"], "beds": p["beds"], "baths": p["baths"]})


def get_property_availability(property_id: str) -> str:
    """Return the listing status (available/pending/sold)."""
    p = _PROPERTIES.get(property_id)
    if not p:
        return json.dumps({"error": f"unknown property_id {property_id}"})
    return json.dumps({"property_id": property_id, "status": p["status"]})


def get_property_size(property_id: str) -> str:
    """Return the square footage of a property."""
    p = _PROPERTIES.get(property_id)
    if not p:
        return json.dumps({"error": f"unknown property_id {property_id}"})
    return json.dumps({"property_id": property_id, "sqft": p["sqft"]})


def get_property_price(property_id: str) -> str:
    """Return the listed price of a property."""
    p = _PROPERTIES.get(property_id)
    if not p:
        return json.dumps({"error": f"unknown property_id {property_id}"})
    return json.dumps({"property_id": property_id, "price": p["price"]})


def get_market_value(property_id: str) -> str:
    """Return the appraised market value of a property."""
    p = _PROPERTIES.get(property_id)
    if not p:
        return json.dumps({"error": f"unknown property_id {property_id}"})
    return json.dumps({"property_id": property_id, "market_value": p["market_value"]})


def get_comparable_sales(neighborhood: str) -> str:
    """Return recent comparable sale prices in a neighborhood."""
    n = _NEIGHBORHOOD_STATS.get(neighborhood)
    if not n:
        return json.dumps({"error": f"unknown neighborhood {neighborhood}"})
    return json.dumps({"neighborhood": neighborhood, "comps": n["comps"]})


def get_neighborhood_avg_price(neighborhood: str) -> str:
    """Return the average price per square foot in a neighborhood."""
    n = _NEIGHBORHOOD_STATS.get(neighborhood)
    if not n:
        return json.dumps({"error": f"unknown neighborhood {neighborhood}"})
    return json.dumps({"neighborhood": neighborhood, "avg_price_per_sqft": n["avg_price_per_sqft"]})


def get_client_preferences(client_id: str) -> str:
    """Return a client's stated preferences."""
    c = _CLIENTS.get(client_id)
    if not c:
        return json.dumps({"error": f"unknown client_id {client_id}"})
    return json.dumps({"client_id": client_id, **c})


def match_property_to_client(property_id: str, client_id: str) -> str:
    """Return whether a property matches a client's preferences."""
    p = _PROPERTIES.get(property_id)
    c = _CLIENTS.get(client_id)
    if not p or not c:
        return json.dumps({"error": "unknown property or client"})
    match = (p["price"] <= c["budget"] and p["beds"] >= c["min_beds"] and p["neighborhood"] == c["neighborhood"])
    return json.dumps({"property_id": property_id, "client_id": client_id, "match": "yes" if match else "no"})
