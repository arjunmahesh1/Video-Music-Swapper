"""Publish CLI: simulate (default) or execute a rollout plan's trafficking.

Dry-run is the product here — it prints the exact HTTP request sequence a
platform integration would send, so the deployment mechanism can be tested
locally with zero spend and zero network traffic:

    python -m sonic_segments.publish --job <campaign_id> --platform meta

Live mode exists only for Meta and only against a SANDBOX ad account. It
refuses to run unless you pass --live AND export META_ACCESS_TOKEN,
META_AD_ACCOUNT_ID (sandbox act_...), and META_PAGE_ID. Even then every
entity is created status=PAUSED, so nothing can ever spend. Google Ads /
DV360 / TikTok / Spotify stay dry-run (their bundles are Editor/SDF/UI
uploads by design — see DEPLOYMENT.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

GRAPH = "https://graph.facebook.com/v21.0"
JOBS_ROOT = Path("output/campaigns")


# ---------------------------------------------------------------------------
# Request planning (pure: no network, fully testable)
# ---------------------------------------------------------------------------

def build_meta_requests(plan: dict[str, Any], ad_account: str = "act_<AD_ACCOUNT_ID>",
                        page_id: str = "<PAGE_ID>") -> list[dict[str, Any]]:
    """The exact Graph API call sequence for this plan, with {placeholders}
    that live mode resolves from each response's returned id."""
    units = [u for u in plan["ad_units"] if u["platform"] == "meta"]
    if not units:
        return []
    steps: list[dict[str, Any]] = [{
        "step": "campaign",
        "method": "POST",
        "url": f"{GRAPH}/{ad_account}/campaigns",
        "body": {
            "name": f"SS_{plan['brand']}_{plan['campaign_id'][:8]}".upper(),
            "objective": "OUTCOME_AWARENESS",
            "special_ad_categories": [],
            "status": "PAUSED",
        },
        "yields": "campaign_id",
    }]
    for i, u in enumerate(units):
        spec = u["platform_spec"]
        steps.append({
            "step": f"video_{i}",
            "method": "POST",
            "url": f"{GRAPH}/{ad_account}/advideos",
            "body": {"source": f"@rollout/assets/{Path(u['variant_file']).name}"},
            "yields": f"video_id_{i}",
        })
        steps.append({
            "step": f"adset_{i}",
            "method": "POST",
            "url": f"{GRAPH}/{ad_account}/adsets",
            "body": {
                "name": u["name"],
                "campaign_id": "{campaign_id}",
                "daily_budget": int(round(u["budget"] * 100)),
                "targeting": spec["targeting"],
                "optimization_goal": spec["optimization_goal"],
                "billing_event": spec["billing_event"],
                "bid_strategy": "LOWEST_COST_WITHOUT_CAP",
                "status": "PAUSED",
            },
            "yields": f"adset_id_{i}",
        })
        steps.append({
            "step": f"creative_{i}",
            "method": "POST",
            "url": f"{GRAPH}/{ad_account}/adcreatives",
            "body": {
                "name": f"{u['name']}_CREATIVE",
                "object_story_spec": {
                    "page_id": page_id,
                    "video_data": {"video_id": f"{{video_id_{i}}}", "message": u["why"][:200]},
                },
            },
            "yields": f"creative_id_{i}",
        })
        steps.append({
            "step": f"ad_{i}",
            "method": "POST",
            "url": f"{GRAPH}/{ad_account}/ads",
            "body": {
                "name": f"{u['name']}_AD",
                "adset_id": f"{{adset_id_{i}}}",
                "creative": {"creative_id": f"{{creative_id_{i}}}"},
                "status": "PAUSED",
            },
            "yields": f"ad_id_{i}",
        })
    return steps


def dry_run_report(plan: dict[str, Any], platform: str) -> str:
    lines = [f"DRY RUN — {platform} — nothing is sent anywhere.", ""]
    if platform == "meta":
        steps = build_meta_requests(plan)
        if not steps:
            return "No Meta ad units in this plan."
        for s in steps:
            lines.append(f"[{s['step']}] {s['method']} {s['url']}")
            lines.append(json.dumps(s["body"], indent=2))
            lines.append(f"  -> response id captured as {{{s['yields']}}}")
            lines.append("")
        lines.append(f"{len(steps)} requests total; every entity status=PAUSED.")
    elif platform in {"google_ads", "dv360", "tiktok", "spotify"}:
        units = [u for u in plan["ad_units"] if u["platform"] == platform]
        exports = {
            "google_ads": "rollout/google_ads/demand_gen.csv — import via Google Ads Editor (campaigns land Paused)",
            "dv360": "rollout/dv360/sdf_line_items.csv — SDF upload from the advertiser's insertion-order page (line items land Draft)",
            "tiktok": "rollout/tiktok/adgroups.json — POST /open_api/v1.3/adgroup/create/ per entry (operation_status=DISABLE)",
            "spotify": "rollout/spotify/adsets.json + audio/ — self-serve upload in Spotify Ads Manager (drafts)",
        }
        lines.append(f"{len(units)} ad unit(s) for {platform}.")
        lines.append(f"Deployment path: {exports[platform]}")
    else:
        return f"Unknown platform: {platform}"
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Live execution (Meta sandbox only; opt-in twice)
# ---------------------------------------------------------------------------

def execute_meta_live(plan: dict[str, Any], job_dir: Path) -> None:
    import requests

    token = os.environ.get("META_ACCESS_TOKEN")
    account = os.environ.get("META_AD_ACCOUNT_ID")
    page_id = os.environ.get("META_PAGE_ID")
    if not (token and account and page_id):
        sys.exit("Live mode needs META_ACCESS_TOKEN, META_AD_ACCOUNT_ID (sandbox), META_PAGE_ID.")
    if not account.startswith("act_"):
        account = f"act_{account}"

    ids: dict[str, str] = {}
    for step in build_meta_requests(plan, ad_account=account, page_id=page_id):
        body = json.loads(json.dumps(step["body"]))  # deep copy
        body = _resolve_placeholders(body, ids)
        files = None
        if step["step"].startswith("video_"):
            asset = job_dir / "rollout" / "assets" / Path(str(body.pop("source"))).name.lstrip("@")
            files = {"source": open(asset, "rb")}
            data = {"access_token": token}
        else:
            data = {"access_token": token, **{k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in body.items()}}
        print(f"[live] {step['method']} {step['url']} ({step['step']})")
        resp = requests.post(step["url"], data=data, files=files, timeout=120)
        if files:
            files["source"].close()
        payload = resp.json()
        if "id" not in payload:
            sys.exit(f"Meta API error at {step['step']}: {payload}")
        ids[step["yields"]] = payload["id"]
        print(f"  -> {step['yields']} = {payload['id']}")
    print(f"Done: {len(ids)} entities created, all PAUSED, in {account}.")


def _resolve_placeholders(obj: Any, ids: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, ids) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(v, ids) for v in obj]
    if isinstance(obj, str) and obj.startswith("{") and obj.endswith("}"):
        return ids.get(obj[1:-1], obj)
    return obj


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Simulate or execute a rollout plan.")
    parser.add_argument("--job", required=True, help="campaign job id (folder under output/campaigns)")
    parser.add_argument("--platform", default="meta",
                        choices=["meta", "google_ads", "dv360", "tiktok", "spotify"])
    parser.add_argument("--live", action="store_true",
                        help="actually create PAUSED entities (Meta sandbox only; needs env credentials)")
    args = parser.parse_args(argv)

    job_dir = JOBS_ROOT / args.job
    plan_path = job_dir / "rollout" / "rollout_plan.json"
    if not plan_path.exists():
        sys.exit(f"No rollout plan at {plan_path} — build one from the preview page or the API first.")
    plan = json.loads(plan_path.read_text())

    if plan.get("issues"):
        print("Plan gate-check issues (fix before going live):")
        for issue in plan["issues"]:
            print(f"  - {issue}")
        print()

    if not args.live:
        print(dry_run_report(plan, args.platform))
        return

    if args.platform != "meta":
        sys.exit(f"Live mode is Meta-sandbox-only; {args.platform} deploys via its bundle file (see DEPLOYMENT.md).")
    if plan.get("issues"):
        sys.exit("Refusing live mode while the plan has gate-check issues.")
    execute_meta_live(plan, job_dir)


if __name__ == "__main__":
    main()
