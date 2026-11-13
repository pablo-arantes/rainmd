#!/usr/bin/env bash
#
# Register this machine as a self-hosted runner for the `gpu` workflow.
#
# 🔴 READ THIS FIRST: THIS REPOSITORY IS PUBLIC, AND A SELF-HOSTED RUNNER ON A
# PUBLIC REPOSITORY IS A REAL EXPOSURE. A workflow is code. If any workflow that
# targets this runner can be triggered from a fork's pull request, then anyone
# who can open one can run whatever they like on this machine, as your user.
# GitHub's own guidance is to use self-hosted runners with private repositories
# only.
#
# `.github/workflows/gpu.yml` is written so that path does not exist: it
# triggers on push-to-main, a nightly schedule and manual dispatch, and NOT on
# pull_request. This script refuses to install if that ever stops being true,
# because the check is worth more than the comment.
#
# Defence in depth, worth turning on either way:
#   Settings -> Actions -> General -> "Require approval for all external
#   contributors". The repository currently allows first-time-contributor
#   approval only, so a returning contributor is not prompted.
#
# Usage:  bash tools/setup-gpu-runner.sh --check-only   # interlocks only
#         bash tools/setup-gpu-runner.sh --yes          # install
# Remove: cd ~/actions-runner-rainmd && ./svc.sh stop && ./svc.sh uninstall \
#           && ./config.sh remove --token "$(gh api -X POST \
#              repos/OWNER/REPO/actions/runners/remove-token --jq .token)"
set -euo pipefail

VERSION=2.338.2
SHA256=25a065af409ffa317ab4d5a3b614dd15fc9c40f2172bc1b35a1b1258e919a388
PACKAGE="actions-runner-osx-arm64-${VERSION}.tar.gz"
TARGET="$HOME/actions-runner-rainmd"
REPO=$(git config --get remote.origin.url | sed -E 's#.*github\.com[:/]([^/]+/[^/.]+)(\.git)?#\1#')
WORKFLOW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.github/workflows/gpu.yml"

if [ "${1:-}" != "--yes" ] && [ "${1:-}" != "--check-only" ]; then
  cat <<TXT
This will, on $REPO:
  - download the GitHub Actions runner v$VERSION into $TARGET
  - verify it against GitHub's published SHA-256 and refuse on a mismatch
  - register it with the label 'webgpu'
  - install it as a launchd service that starts at login and listens for jobs

That service executes code from the repository on this machine. Read the notes
at the top of this file before running it, then re-run with --yes.
TXT
  exit 1
fi

# 🔴 THE INTERLOCKS, AND THEY ARE CHECKED AGAINST THE ACTIVE YAML ONLY. The
# first version of this grepped the whole file for 'self-hosted' and so matched
# its own header prose - the check passed on a workflow that had been changed to
# ubuntu-latest, which is the exact failure mode of a safety check that reads
# documentation instead of configuration. Comments are stripped first.
active() { sed -E 's/#.*$//' "$WORKFLOW"; }

if active | grep -qE '^[[:space:]]*pull_request[[:space:]]*:'; then
  echo "REFUSING: $WORKFLOW has a pull_request trigger." >&2
  echo "On a public repository that lets a fork run code on this machine." >&2
  exit 1
fi
if ! active | grep -qE '^[[:space:]]*runs-on:.*self-hosted'; then
  echo "REFUSING: no job in $WORKFLOW has a self-hosted runs-on, so this" >&2
  echo "installation would do nothing." >&2
  exit 1
fi
echo "ok: every runs-on is self-hosted and there is no pull_request trigger"

# 🔴 AND THE CHECKS CAN BE RUN WITHOUT INSTALLING ANYTHING, because the first
# time they were exercised the only way to reach them was to run the script for
# real - which downloaded 128 MB and wrote a .env before a closed pipe happened
# to stop it. A safety check you can only test by doing the dangerous thing is
# not a safety check.
if [ "${1:-}" = "--check-only" ]; then
  echo "--check-only: interlocks pass, nothing installed"
  exit 0
fi

# 🔴 ITS OWN EMPTY DIRECTORY, AND NEVER AN EXISTING ONE. The tarball is
# downloaded code; extracting it over something else is how a surprise happens.
if [ -e "$TARGET" ]; then
  echo "REFUSING: $TARGET already exists. Remove it first, or finish the" >&2
  echo "existing installation." >&2
  exit 1
fi
mkdir -p "$TARGET"
cd "$TARGET"

echo "downloading $PACKAGE"
curl -fsSL -O "https://github.com/actions/runner/releases/download/v${VERSION}/${PACKAGE}"
got=$(shasum -a 256 "$PACKAGE" | cut -d' ' -f1)
if [ "$got" != "$SHA256" ]; then
  echo "CHECKSUM MISMATCH - refusing to extract" >&2
  echo "  expected $SHA256" >&2
  echo "  got      $got" >&2
  exit 1
fi
echo "checksum matches GitHub's published value"
tar xzf "$PACKAGE"

# 🔴 A launchd SERVICE GETS A MINIMAL PATH, WHICH IS HOW THESE LANES FAIL FIRST.
# node, npm and the python that carries OpenMM and MDAnalysis are not on it.
# The runner reads this file for every job, so the lanes see the same toolchain
# an interactive shell does.
NODE_DIR=$(dirname "$(command -v node)")
PY_DIR=$(dirname "$(command -v python3)")
printf 'PATH=%s:%s:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin\n' "$NODE_DIR" "$PY_DIR" > .env
echo "wrote .env with PATH including $NODE_DIR and $PY_DIR"

echo "requesting a registration token for $REPO"
TOKEN=$(gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token)

./config.sh --unattended \
  --url "https://github.com/${REPO}" \
  --token "$TOKEN" \
  --name "$(scutil --get LocalHostName 2>/dev/null || hostname)-webgpu" \
  --labels webgpu \
  --work _work \
  --replace

./svc.sh install
./svc.sh start
echo
echo "installed and started. Check it with:"
echo "  gh api repos/${REPO}/actions/runners --jq '.runners[] | \"\\(.name) \\(.status) \\(.labels[].name)\"'"
echo "Then run the lanes with:"
echo "  gh workflow run gpu"
echo
echo "To remove it again, see the header of this file."
