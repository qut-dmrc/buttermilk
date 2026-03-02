import os
import argparse
import subprocess
import sys
from typing import Optional

try:
    from packaging.version import Version, InvalidVersion
except ImportError:
    print("Error: 'packaging' library not found. Install it with 'pip install packaging'.")
    sys.exit(1)

def run_git(args: list[str]) -> Optional[str]:
    """Run a git command and return the output."""
    try:
        return subprocess.check_output(["git"] + args, stderr=subprocess.PIPE).decode().strip()
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="replace") if e.stderr is not None else ""
        # Only treat known "no tags" / "no exact match" errors as non-fatal.
        if "No names found" in stderr or "no tag exactly matches" in stderr:
            return None
        # For all other git errors, re-raise so CI fails loudly instead of tagging from a bad state.
        raise

def get_latest_tag() -> Optional[str]:
    """Get the latest tag from git."""
    return run_git(["describe", "--tags", "--abbrev=0"])

def is_current_commit_tagged() -> Optional[str]:
    """Check if the current commit is already tagged."""
    return run_git(["describe", "--tags", "--exact-match"])

def bump_version(current_ver: Version, branch: str) -> str:
    """Calculate the next version based on branch."""
    if branch == "stable":
        # Bump minor: 0.6.1 -> 0.7.0
        return f"{current_ver.major}.{current_ver.minor + 1}.0"
    elif branch == "dev":
        # Bump patch: 0.6.1 -> 0.6.2
        return f"{current_ver.major}.{current_ver.minor}.{current_ver.micro + 1}"
    else:
        raise ValueError(f"Branch '{branch}' not configured for auto-bump.")

def main():
    parser = argparse.ArgumentParser(description="Bump version based on branch.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing.")
    args = parser.parse_args()

    # Determine current branch
    branch = os.environ.get("GITHUB_REF_NAME")
    if not branch:
        # Fallback for local testing if not set
        try:
            branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        except Exception:
            branch = "unknown"

    # If running locally without GITHUB_REF_NAME, allow manual override via env var
    if not branch or branch == "HEAD":
        branch = os.environ.get("MANUAL_BRANCH_OVERRIDE", branch)

    print(f"Current branch: {branch}")

    if branch not in ["dev", "stable"]:
        print(f"Skipping version bump for branch '{branch}'. Only 'dev' and 'stable' are supported.")
        # Exit with 0 as this is not an error condition for the workflow (just skip)
        sys.exit(0)

    # Check if already tagged
    existing_tag = is_current_commit_tagged()
    if existing_tag:
        print(f"Commit is already tagged as '{existing_tag}'. Skipping bump.")
        sys.exit(0)

    latest_tag = get_latest_tag()
    if latest_tag:
        print(f"Latest tag found: {latest_tag}")
        try:
            # handle 'v' prefix
            clean_ver = latest_tag.lstrip("v")
            current_ver = Version(clean_ver)
        except InvalidVersion:
            print(f"Error: Latest tag '{latest_tag}' is not a valid version.")
            sys.exit(1)
    else:
        print("No tags found. Starting at 0.6.1")
        current_ver = Version("0.6.1")

    try:
        new_ver_str = bump_version(current_ver, branch)
        new_tag = f"v{new_ver_str}"
        print(f"Bumping version: {current_ver} -> {new_ver_str} ({new_tag})")
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    if args.dry_run:
        print(f"[DRY RUN] Would execute: git tag -a {new_tag} -m 'Auto-bump to {new_tag}'")
        print(f"[DRY RUN] Would execute: git push origin {new_tag}")
    else:
        # Configure git user if running in CI
        if os.environ.get("GITHUB_ACTIONS"):
            try:
                subprocess.run(["git", "config", "user.name", "github-actions[bot]"], check=True)
                subprocess.run(["git", "config", "user.email", "github-actions[bot]@users.noreply.github.com"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error configuring git user: {e}")
                sys.exit(1)

        print(f"Creating tag {new_tag}...")
        try:
            subprocess.run(["git", "tag", "-a", new_tag, "-m", f"Auto-bump to {new_tag}"], check=True)
            print(f"Pushing tag {new_tag}...")
            subprocess.run(["git", "push", "origin", new_tag], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating/pushing tag: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
