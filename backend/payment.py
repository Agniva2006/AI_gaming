import os
import json
import stripe
import sys
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends, Header, status
from pydantic import BaseModel
from typing import Optional

# Ensure parent directory is in sys.path to resolve backend package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.auth import load_users, save_users, PLANS, get_current_user

router = APIRouter(prefix="/payment", tags=["billing"])

# Initialize Stripe API configurations
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY
    IS_SANDBOX = False
else:
    # Sandbox mode if keys aren't configured
    IS_SANDBOX = True

# Stripe Price IDs mapping
STRIPE_PRICES = {
    "pro": os.environ.get("STRIPE_PRICE_PRO", "price_mock_pro_trainer_99"),
    "developer": os.environ.get("STRIPE_PRICE_DEVELOPER", "price_mock_developer_499")
}

class CheckoutRequest(BaseModel):
    plan: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

class PortalRequest(BaseModel):
    return_url: Optional[str] = None

@router.post("/checkout")
async def create_checkout_session(req: CheckoutRequest, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    plan = req.plan.lower().strip()
    email = current_user.get("email")
    fullname = current_user.get("full_name", "")
    
    if plan not in PLANS:
        raise HTTPException(status_code=400, detail="Invalid subscription plan selected.")
        
    if plan == current_user.get("plan"):
        return {"success": False, "message": "You are already subscribed to this plan."}

    # Handle Sandbox Emulation
    if IS_SANDBOX:
        import secrets
        session_id = f"cs_sandbox_" + secrets.token_hex(16)
        mock_success = req.success_url or f"/frontend/index.html?payment=success&session_id={session_id}&plan={plan}"
        return {
            "success": True,
            "session_id": session_id,
            "checkout_url": mock_success,
            "sandbox": True,
            "message": "Checkout session created (Sandbox mode)."
        }

    # Real Stripe Implementation
    try:
        users = load_users()
        u = users[username]
        customer_id = u.get("stripe_customer_id")
        
        if not customer_id:
            customer = stripe.Customer.create(
                email=email,
                name=fullname,
                metadata={"username": username}
            )
            customer_id = customer.id
            u["stripe_customer_id"] = customer_id
            save_users(users)

        price_id = STRIPE_PRICES.get(plan)
        if not price_id:
            raise HTTPException(status_code=500, detail=f"Price ID configuration missing for plan: {plan}")

        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            mode="subscription",
            success_url=req.success_url or "http://127.0.0.1:8000/frontend/index.html?payment=success&session_id={CHECKOUT_SESSION_ID}",
            cancel_url=req.cancel_url or "http://127.0.0.1:8000/frontend/index.html?payment=cancelled",
            metadata={
                "username": username,
                "plan": plan
            },
            client_reference_id=username
        )
        
        return {
            "success": True,
            "session_id": session.id,
            "checkout_url": session.url,
            "sandbox": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe Session Error: {str(e)}")

@router.post("/portal")
async def create_portal_session(req: PortalRequest, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    customer_id = current_user.get("stripe_customer_id")
    
    if IS_SANDBOX:
        return {
            "success": True,
            "portal_url": "/frontend/index.html?payment=portal",
            "sandbox": True
        }
        
    if not customer_id:
        raise HTTPException(
            status_code=400, 
            detail="No Stripe customer record found. Subscribe to a plan first."
        )
        
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=req.return_url or "http://127.0.0.1:8000/frontend/index.html"
        )
        return {
            "success": True,
            "portal_url": session.url,
            "sandbox": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe Portal Error: {str(e)}")

@router.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    payload = await request.body()
    
    if IS_SANDBOX or not stripe_signature or not STRIPE_WEBHOOK_SECRET:
        try:
            event_dict = json.loads(payload.decode("utf-8"))
            event = stripe.Event.construct_from(event_dict, key=None)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON Parse Error: {str(e)}")
    else:
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, STRIPE_WEBHOOK_SECRET
            )
        except stripe.error.SignatureVerificationError as e:
            raise HTTPException(status_code=400, detail="Invalid webhook signature.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Webhook structure error: {str(e)}")

    event_type = event.type
    session_data = event.data.object
    
    if event_type == "checkout.session.completed":
        username = session_data.get("client_reference_id") or session_data.get("metadata", {}).get("username")
        plan = session_data.get("metadata", {}).get("plan")
        customer_id = session_data.get("customer")
        subscription_id = session_data.get("subscription")
        
        if username and plan:
            users = load_users()
            if username in users:
                u = users[username]
                u["plan"] = plan
                u["stripe_customer_id"] = customer_id
                u["stripe_subscription_id"] = subscription_id
                u["subscription_status"] = "active"
                
                activity = u.setdefault("activity", [])
                activity.append({
                    "endpoint": f"Stripe Checkout completed ({plan})",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                save_users(users)
                
    elif event_type == "invoice.paid":
        customer_id = session_data.get("customer")
        if customer_id:
            users = load_users()
            for un, ud in users.items():
                if ud.get("stripe_customer_id") == customer_id:
                    ud["subscription_status"] = "active"
                    activity = ud.setdefault("activity", [])
                    activity.append({
                        "endpoint": "Stripe Invoice paid - Subscription renewed",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    save_users(users)
                    break
                    
    elif event_type in ["invoice.payment_failed", "customer.subscription.deleted"]:
        customer_id = session_data.get("customer")
        if customer_id:
            users = load_users()
            for un, ud in users.items():
                if ud.get("stripe_customer_id") == customer_id:
                    old_plan = ud.get("plan", "free")
                    ud["plan"] = "free"
                    ud["subscription_status"] = "canceled"
                    ud["stripe_subscription_id"] = ""
                    
                    activity = ud.setdefault("activity", [])
                    activity.append({
                        "endpoint": f"Stripe Subscription canceled (reverted {old_plan} -> free)",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    save_users(users)
                    break

    return {"status": "success"}
