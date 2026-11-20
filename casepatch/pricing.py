"""Local quote for a proposed tariff. This neither meters an account nor bills it."""
from decimal import Decimal, ROUND_HALF_UP

from .core import PolicyError, walk

FREE_MONTHLY_CHARACTERS = 100_000
USD_PER_MILLION_CHARACTERS = Decimal("10")


def estimate(records, monthly_used_characters=0):
    if not isinstance(records, list):
        raise PolicyError("RECORDS_MUST_BE_LIST")
    if isinstance(monthly_used_characters, bool) or not isinstance(monthly_used_characters, int) or monthly_used_characters < 0:
        raise PolicyError("INVALID_MONTHLY_USAGE")
    characters = sum(len(value) for _, value in walk(records))
    used_free = min(monthly_used_characters, FREE_MONTHLY_CHARACTERS)
    allowance = FREE_MONTHLY_CHARACTERS - used_free
    billable = max(0, characters - allowance)
    exact = Decimal(billable) * USD_PER_MILLION_CHARACTERS / Decimal(1_000_000)
    return {
        "pricing_status": "PROPOSED_NOT_LIVE",
        "source_characters": characters,
        "meter": "Unicode code points in record string values; keys and policy annotations excluded",
        "assumed_monthly_used_characters": monthly_used_characters,
        "free_characters_applied": min(characters, allowance),
        "estimated_billable_characters": billable,
        "rate_usd_per_million_characters": str(USD_PER_MILLION_CHARACTERS),
        "estimated_usage_charge_usd_unrounded": str(exact),
        "illustrative_charge_usd_rounded": str(exact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        "actual_charge_usd": "0.00",
        "note": "Free local estimate. No account queried or charged. Monthly invoices would round the aggregate once.",
    }
