# app/profiles.py
import os
import json
from typing import List, Optional
from .config_types import UserProfile
from .io_utils import ensure_dir

PROFILE_DIR = os.path.join("configs", "profiles")

def list_profiles() -> List[str]:
    """List available profile names."""
    ensure_dir(PROFILE_DIR)
    files = [f for f in os.listdir(PROFILE_DIR) if f.endswith(".json")]
    return [f.replace(".json", "") for f in files]

def save_profile(profile: UserProfile):
    """Save a user profile to disk."""
    ensure_dir(PROFILE_DIR)
    path = os.path.join(PROFILE_DIR, f"{profile.name}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(profile.model_dump_json(indent=2))

def load_profile(name: str) -> Optional[UserProfile]:
    """Load a user profile from disk."""
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return UserProfile(**data)

def delete_profile(name: str):
    """Delete a profile."""
    path = os.path.join(PROFILE_DIR, f"{name}.json")
    if os.path.exists(path):
        os.remove(path)
