from __future__ import annotations

import os
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()


@router.post("/billing/stripe/webhook")
async def stripe_webhook(request: Request):
    """Stripe webhook endpoint (optional). Verifies signature if secret set.
    This is a stub to record events and could be extended to manage subscriptions.
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
    try:
        import stripe  # type: ignore
        if secret:
            event = stripe.Webhook.construct_event(payload, sig_header, secret)
        else:
            event = request.json()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))

    # TODO: map event types to user plan/usage. For now just acknowledge.
    return {"received": True}

