#!/usr/bin/env python3
"""
Environment setup script for IMDC2026 / Proteus project.
Run once after cloning the repository on a new machine:

    python setup_env.py

Add new steps by appending Step instances to the STEPS list at the bottom.
"""

import subprocess
import sys
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Step primitives
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """A single setup step."""
    name: str
    cmd: list[str] | None = None                  # static command, or …
    build_cmd: Callable[[], list[str]] | None = None  # … a callable that returns one
    prompts: list[dict] = field(default_factory=list)
    # prompts items: {"key": str, "message": str}  – collected before build_cmd is called

    def run(self) -> bool:
        """Execute the step. Returns True on success."""
        print(f"\n{'─' * 60}")
        print(f"  {self.name}")
        print(f"{'─' * 60}")

        # Collect any required user input
        answers: dict[str, str] = {}
        for prompt in self.prompts:
            value = input(f"  {prompt['message']}: ").strip()
            answers[prompt["key"]] = value

        # Resolve the command
        if self.build_cmd is not None:
            cmd = self.build_cmd(**answers)
        elif self.cmd is not None:
            cmd = self.cmd
        else:
            raise ValueError(f"Step '{self.name}' has neither cmd nor build_cmd.")

        print(f"  $ {' '.join(cmd)}\n")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\n  ✗ Step failed (exit code {result.returncode}).")
            return False

        print(f"\n  ✓ Done.")
        return True


# ---------------------------------------------------------------------------
# Step definitions  ←  add / remove / reorder here
# ---------------------------------------------------------------------------

def _dvc_remote_user(username: str) -> list[str]:
    if username:
        return [
            "uv", "run", "dvc", "remote", "modify", "--local",
            "la_berenjena", "user", username,
        ]
    else:
        return (
            "uv run dvc config --local core.remote la_berenjena_local".split()
        )


STEPS: list[Step] = [
    Step(
        name="Sync Python environment  (uv sync --extra dev)",
        cmd=["uv", "sync", "--extra", "dev"],
    ),
    Step(
        name="Enable nbdime git integration",
        cmd=["uv", "run", "nbdime", "config-git", "--enable"],
    ),
    Step(
        name="Configure DVC remote user for 'la_berenjena'",
        prompts=[{
            "key": "username",
            "message": (
                "Enter your username on la_berenjena server, or leave blank if "
                "you are running directly on the server.")
        }],
        build_cmd=_dvc_remote_user,
    ),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n╔══════════════════════════════════════════════╗")
    print("║   InfraMIND / Proteus  –  Environment Setup  ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  {len(STEPS)} step(s) will be executed.\n")

    failed: list[str] = []
    for step in STEPS:
        ok = step.run()
        if not ok:
            failed.append(step.name)

    print(f"\n{'═' * 60}")
    if failed:
        print(f"  Setup finished with {len(failed)} error(s): ❌")
        for name in failed:
            print(f"    ✗ {name}")
        print(f"{'═' * 60}\n")
        sys.exit(1)
    else:
        print("  All steps completed successfully. ✅")
        print(f"{'═' * 60}")

        # Extra
        print(" - To activate the environment, run: `source .venv/bin/activate`")
        print(" - To locally cache all large data files, run: `dvc pull`")
        print()


if __name__ == "__main__":
    main()

