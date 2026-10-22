# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Mock order and billing systems for the spec-based test environments."""

_ORDERS = {
    "ORD-1001": {"status": "shipped", "carrier": "DHL"},
    "ORD-1002": {"status": "processing", "carrier": "UPS"},
}
_INVOICES = {
    "INV-2001": {"total": "149.90 EUR", "payment": "paid"},
    "INV-2002": {"total": "612.00 EUR", "payment": "overdue"},
}


def get_order(order_id: str) -> str:
    """Look up an order by its ID."""
    return str(_ORDERS.get(order_id, {"error": f"unknown order {order_id}"}))


def get_invoice(invoice_id: str) -> str:
    """Look up an invoice by its ID."""
    return str(_INVOICES.get(invoice_id, {"error": f"unknown invoice {invoice_id}"}))
