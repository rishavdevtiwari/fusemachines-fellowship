"""
External Tools & Function Calling Implementations for ShopAssist AI (Week 16).
Provides domain operations for e-commerce orders, cancellation fee calculation,
refund eligibility assessment, courier verification, human escalation, and RAG search.
Includes simulation switches for Failure Injection Testing (Task 3 Requirement).
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from src.config import settings

def _load_orders() -> List[Dict[str, Any]]:
    """Loads sample order records from disk."""
    if not os.path.exists(settings.orders_file):
        return []
    try:
        with open(settings.orders_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("orders", [])
    except Exception as e:
        print(f"Error reading orders file: {e}")
        return []

def check_order_status(order_id: str) -> Dict[str, Any]:
    """
    Retrieves real-time status, courier tracking, items, and billing details for an order.
    """
    # Failure Injection Hook
    if settings.failure_simulation_mode == "tool_unavailable":
        return {
            "status": "simulated_failure",
            "error_type": "DatabaseConnectionTimeout",
            "error_message": "SQL Connection pool exhausted: Warehouse DB cluster unavailable (HTTP 503)."
        }
    if settings.failure_simulation_mode == "timeout":
        return {
            "status": "simulated_failure",
            "error_type": "OperationTimeout",
            "error_message": "Gateway timeout waiting for order management upstream (HTTP 504)."
        }
    if settings.failure_simulation_mode == "malformed_output":
        return {
            "status": "success",
            "raw_corrupted_payload": "ERR_NULL_POINTER_EXCEPTION: [0xDEADBEEF] Corrupted binary stream"
        }

    clean_id = order_id.strip().upper()
    orders = _load_orders()
    for order in orders:
        if order.get("order_id") == clean_id:
            return {
                "status": "success",
                "order_id": clean_id,
                "customer_name": order.get("customer_name"),
                "order_status": order.get("status"),
                "shipping_tier": order.get("shipping_tier"),
                "courier": order.get("courier"),
                "tracking_number": order.get("tracking_number"),
                "estimated_delivery": order.get("estimated_delivery", "N/A"),
                "total_amount": order.get("total_amount"),
                "items": order.get("items"),
                "shipping_address": order.get("shipping_address"),
                "hours_since_order": order.get("hours_since_order", 12.0),
                "days_since_delivery": order.get("days_since_delivery", 0)
            }
            
    return {
        "status": "not_found",
        "order_id": clean_id,
        "message": f"Order '{clean_id}' was not found in the order management database."
    }

def calculate_cancellation_fee(order_id: str, hours_elapsed: Optional[float] = None) -> Dict[str, Any]:
    """
    Calculates the applicable cancellation penalty fee based on elapsed order processing time.
    """
    clean_id = order_id.strip().upper()
    orders = _load_orders()
    order_data = next((o for o in orders if o.get("order_id") == clean_id), None)
    
    if hours_elapsed is None:
        if order_data and "hours_since_order" in order_data:
            hours_elapsed = float(order_data["hours_since_order"])
        else:
            hours_elapsed = 2.0
            
    total_amount = order_data.get("total_amount", 100.0) if order_data else 100.0
    order_status = order_data.get("status", "PROCESSING") if order_data else "PROCESSING"
    
    if order_status in ["SHIPPED", "DELIVERED"]:
        return {
            "status": "cannot_cancel_direct",
            "order_id": clean_id,
            "order_status": order_status,
            "hours_elapsed": hours_elapsed,
            "applicable_fee": 0.0,
            "can_cancel": False,
            "message": (
                f"Order {clean_id} has already reached status '{order_status}' and cannot be cancelled directly. "
                "The customer may refuse delivery or request a return under our Return Policy."
            )
        }
        
    if hours_elapsed <= 1.0:
        fee = 0.0
        stage = "Grace Period (0-1 hour)"
        notes = "Free cancellation with 100% full refund."
    elif hours_elapsed <= 6.0:
        fee = 5.00
        stage = "Processing & Staging (1-6 hours)"
        notes = "Standard $5.00 restocking fee applies."
    elif hours_elapsed <= 24.0:
        fee = 15.00
        stage = "Picking & Boxing (6-24 hours)"
        notes = "$15.00 warehouse recovery and restocking fee applies."
    else:
        return {
            "status": "cannot_cancel_direct",
            "order_id": clean_id,
            "order_status": order_status,
            "hours_elapsed": hours_elapsed,
            "can_cancel": False,
            "message": f"Order {clean_id} exceeded the 24-hour cancellation window ({hours_elapsed:.1f} hours elapsed)."
        }
        
    net_refund = max(0.0, total_amount - fee)
    return {
        "status": "success",
        "order_id": clean_id,
        "hours_elapsed": hours_elapsed,
        "stage": stage,
        "applicable_fee": fee,
        "total_order_amount": total_amount,
        "estimated_net_refund": round(net_refund, 2),
        "can_cancel": True,
        "notes": notes
    }

def check_refund_eligibility(
    order_id: str,
    days_since_delivery: Optional[int] = None,
    item_opened: bool = False,
    damage_reported: bool = False
) -> Dict[str, Any]:
    """
    Evaluates customer refund eligibility against standard corporate return policies,
    accounting for damage waivers and item condition.
    """
    clean_id = order_id.strip().upper()
    orders = _load_orders()
    order_data = next((o for o in orders if o.get("order_id") == clean_id), None)
    
    if days_since_delivery is None:
        if order_data and "days_since_delivery" in order_data:
            days_since_delivery = int(order_data["days_since_delivery"])
        else:
            days_since_delivery = 10
            
    total_amount = order_data.get("total_amount", 99.99) if order_data else 99.99
    
    if days_since_delivery > 30:
        return {
            "status": "ineligible",
            "order_id": clean_id,
            "eligible": False,
            "reason": f"Standard return window expired ({days_since_delivery} days elapsed; policy limit is 30 calendar days).",
            "days_since_delivery": days_since_delivery,
            "total_amount": total_amount
        }
        
    # Standard shipping deduction is $5.99 unless damaged on arrival
    shipping_deduction = 0.0 if damage_reported else 5.99
    net_refund = max(0.0, total_amount - shipping_deduction)
    
    return {
        "status": "eligible",
        "order_id": clean_id,
        "eligible": True,
        "days_since_delivery": days_since_delivery,
        "item_opened": item_opened,
        "damage_reported": damage_reported,
        "total_amount": total_amount,
        "shipping_deduction": shipping_deduction,
        "fee_waived": damage_reported,
        "estimated_refund": round(net_refund, 2),
        "instructions": (
            "Damage claim accepted: Prepaid label provided with $0 return fee."
            if damage_reported else
            "Prepaid return label generated. Flat $5.99 return shipping deduction applies."
        )
    }

def verify_courier_tracking(order_id: str, courier: Optional[str] = None, tracking_number: Optional[str] = None) -> Dict[str, Any]:
    """
    Cross-source verification tool: Queries live carrier logistics telemetry (FedEx/UPS)
    to verify delivery status, transit delays, or damaged-in-transit scans.
    """
    clean_id = order_id.strip().upper()
    orders = _load_orders()
    order = next((o for o in orders if o.get("order_id") == clean_id), None)
    
    if not order:
        return {"status": "not_found", "message": f"No shipment tracking record for {clean_id}"}
        
    trk = tracking_number or order.get("tracking_number")
    carrier = courier or order.get("courier", "FedEx")
    status = order.get("status", "IN_TRANSIT")
    
    # Simulate cross-source carrier scan telemetry
    carrier_scans = [
        {"timestamp": "2026-09-02T14:22:00Z", "location": "Origin Sort Facility", "event": "Departed"},
        {"timestamp": "2026-09-03T08:15:00Z", "location": "Regional Hub", "event": "In Transit"}
    ]
    if status == "DELIVERED":
        carrier_scans.append({"timestamp": "2026-09-04T11:30:00Z", "location": "Destination", "event": "Delivered to Porch"})
    
    return {
        "status": "success",
        "order_id": clean_id,
        "courier": carrier,
        "tracking_number": trk,
        "verified_delivery_status": status,
        "latest_scan": carrier_scans[-1]["event"],
        "carrier_telemetry_consistent": True,
        "history": carrier_scans
    }

def escalate_to_human(
    order_id: Optional[str] = None,
    issue_summary: str = "Unresolved customer dispute",
    urgency: str = "normal"
) -> Dict[str, Any]:
    """
    Escalates high-friction, legally sensitive, or unresolvable disputes to a Tier-2 human supervisor.
    """
    ticket_id = f"TICK-{uuid.uuid4().hex[:6].upper()}"
    return {
        "status": "escalated",
        "ticket_id": ticket_id,
        "order_id": order_id,
        "urgency": urgency.upper(),
        "department": "Tier-2 Specialized Customer Resolution",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": issue_summary,
        "estimated_agent_response_time": "15 minutes" if urgency.lower() == "high" else "2 hours",
        "message": f"Case escalated to Tier-2 supervisor. Reference Ticket ID: {ticket_id}."
    }

# Tool Registry
TOOL_REGISTRY = {
    "check_order_status": check_order_status,
    "calculate_cancellation_fee": calculate_cancellation_fee,
    "check_refund_eligibility": check_refund_eligibility,
    "verify_courier_tracking": verify_courier_tracking,
    "escalate_to_human": escalate_to_human
}

# Tool OpenAPI Schema Specifications
TOOL_DEFINITIONS = [
    {
        "name": "check_order_status",
        "description": "Retrieves real-time fulfillment status, courier name, tracking number, item details, and timestamps for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. ORD-1001 or ORD-1002"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "calculate_cancellation_fee",
        "description": "Calculates cancellation penalty fee, refund breakdown, and cancellation eligibility based on order status and elapsed time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to evaluate for cancellation"},
                "hours_elapsed": {"type": "number", "description": "Optional elapsed hours since order was placed"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "check_refund_eligibility",
        "description": "Evaluates return & refund compliance based on delivery date, product condition, and damage waiver exceptions.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID for refund evaluation"},
                "days_since_delivery": {"type": "integer", "description": "Days elapsed since delivery was confirmed"},
                "item_opened": {"type": "boolean", "description": "Whether package has been unsealed or opened"},
                "damage_reported": {"type": "boolean", "description": "Whether item arrived damaged or defective"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "verify_courier_tracking",
        "description": "Cross-verifies shipment status against carrier logistics telemetry (FedEx/UPS) for delivery discrepancies.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID to cross-verify"},
                "courier": {"type": "string", "description": "Courier carrier name (e.g. FedEx, UPS)"},
                "tracking_number": {"type": "string", "description": "Tracking number"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Escalates complex, high-friction, or sensitive inquiries to a Tier-2 human customer care supervisor.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Related order ID if applicable"},
                "issue_summary": {"type": "string", "description": "Detailed summary of customer dispute"},
                "urgency": {"type": "string", "enum": ["low", "normal", "high"], "description": "Urgency tier"}
            },
            "required": ["issue_summary"]
        }
    }
]

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Executes a registered domain tool by name with arguments."""
    func = TOOL_REGISTRY.get(tool_name)
    if not func:
        return {
            "status": "error",
            "error_message": f"Tool '{tool_name}' is not recognized in registry."
        }
    try:
        return func(**arguments)
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Exception executing tool '{tool_name}': {str(e)}"
        }
