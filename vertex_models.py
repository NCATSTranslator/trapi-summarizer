#!/usr/bin/env python
"""Utility function to list available models for provider/region.

Needs to use vertex_creds.

Usage:
    python vertex_models.py [--provider gemini|anthropic|both] [--location us-east5]
"""
import argparse
import httpx
import google.auth.transport.requests
from google import genai

import vertex_creds


def _access_token(creds) -> str:
    """Exchange the SA credential for a short-lived bearer token (network call)."""
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def list_gemini(creds, project: str, location: str) -> list[str]:
    client = genai.Client(vertexai=True, project=project, location=location,
                          credentials=creds)
    names = []
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        # Keep generateContent-capable models (or those that don't declare actions).
        if not actions or "generateContent" in actions:
            names.append(getattr(m, "name", str(m)))
    return names


def _api_host(location: str) -> str:
    """Vertex API host for a location. `global`, `us`, and `eu` are not prefixes."""
    if location == "global":
        return "aiplatform.googleapis.com"
    if location in ("us", "eu"):
        return f"aiplatform.{location}.rep.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def list_anthropic(creds, location: str) -> list[str]:
    """Anthropic models on Vertex are 'publisher models'; the anthropic SDK has no
    list endpoint, so query the Vertex Model Garden REST API directly."""
    token = _access_token(creds)
    url = (f"https://{_api_host(location)}/v1beta1/"
           f"publishers/anthropic/models")
    r = httpx.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30.0)
    r.raise_for_status()
    return [m.get("name", "") for m in r.json().get("publisherModels", [])]


def main():
    p = argparse.ArgumentParser(
        description="List available Vertex models (Gemini + Anthropic) for this SA")
    p.add_argument("--provider", choices=("gemini", "anthropic", "both"), default="both")
    p.add_argument("--location", default=vertex_creds.LOCATION)
    args = p.parse_args()

    creds, project = vertex_creds.load_credentials()
    print(f"project={project} location={args.location}")

    if args.provider in ("gemini", "both"):
        print("\n== Gemini models ==")
        try:
            for name in list_gemini(creds, project, args.location):
                print(f"  {name}")
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")

    if args.provider in ("anthropic", "both"):
        print("\n== Anthropic (Claude) publisher models ==")
        try:
            names = list_anthropic(creds, args.location)
            for name in names or ["(none returned)"]:
                print(f"  {name}")
        except Exception as e:
            print(f"  [error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
