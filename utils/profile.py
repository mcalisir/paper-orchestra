"""Selects the writing profile. Set PAPER_ORCHESTRA_PROFILE=education for education journals."""

import os


def education_profile_active() -> bool:
    return os.getenv("PAPER_ORCHESTRA_PROFILE", "").strip().lower() == "education"
