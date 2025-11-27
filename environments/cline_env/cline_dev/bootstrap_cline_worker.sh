#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CLINE_SRC_DIR=${CLINE_SRC_DIR:-/tmp/nous-cline}
CLINE_REPO_URL=${CLINE_REPO_URL:-https://github.com/NousResearch/cline}
# Use the cline submodule from atropos repo if available
CLINE_SUBMODULE_DIR="${ATROPOS_ROOT:-$ROOT_DIR/../..}/environments/cline_env/cline"
WORKSPACE_ROOT=${WORKSPACE_ROOT:-/workspace}
TASK_BOOTSTRAP_SCRIPT=${TASK_BOOTSTRAP_SCRIPT:-}
PROTOBUS_PORT=${PROTOBUS_PORT:-26040}
HOSTBRIDGE_PORT=${HOSTBRIDGE_PORT:-26041}
USE_C8=${USE_C8:-false}
NODE_OPTIONS=${NODE_OPTIONS:---max-old-space-size=4096}

log() {
  printf '[cline-worker] %s\n' "$*"
}

fetch_cline_repo() {
  # Check if we can use the pre-built cline submodule (much faster)
  if [[ -d "$CLINE_SUBMODULE_DIR/dist-standalone" ]]; then
    log "Using pre-built Cline from submodule at $CLINE_SUBMODULE_DIR"
    # Use the submodule directly - no need to copy
    CLINE_SRC_DIR="$CLINE_SUBMODULE_DIR"
    export CLINE_SRC_DIR
    return 0
  fi

  log "Pre-built Cline not found, falling back to clone/build"
  
  # Skip git-lfs to avoid "command not found" errors - LFS files not needed for build
  export GIT_LFS_SKIP_SMUDGE=1
  
  if [[ -d "$CLINE_SRC_DIR/.git" ]]; then
    log "Updating existing Cline repo at $CLINE_SRC_DIR"
    git -C "$CLINE_SRC_DIR" fetch origin main
    # Clean untracked files and reset before checkout to avoid conflicts
    git -C "$CLINE_SRC_DIR" clean -fd
    git -C "$CLINE_SRC_DIR" reset --hard origin/main
  else
    log "Cloning $CLINE_REPO_URL into $CLINE_SRC_DIR"
    rm -rf "$CLINE_SRC_DIR"
    git clone "$CLINE_REPO_URL" "$CLINE_SRC_DIR"
  fi

  apply_cline_patches
}

apply_cline_patches() {
  local patch_dir="$ROOT_DIR/patches"
  if [[ ! -d "$patch_dir" ]]; then
    return
  fi

  pushd "$CLINE_SRC_DIR" >/dev/null
  for patch in "$patch_dir"/*.patch; do
    [[ -f "$patch" ]] || continue
    log "Applying patch $(basename "$patch")"
    git apply --whitespace=nowarn "$patch" || {
      log "Patch $(basename "$patch") failed to apply; continuing"
    }
  done
  popd >/dev/null
}

build_cline() {
  # Skip build if using pre-built submodule
  if [[ -d "$CLINE_SRC_DIR/dist-standalone" ]]; then
    log "Using pre-built Cline standalone at $CLINE_SRC_DIR/dist-standalone"
    return 0
  fi
  
  pushd "$CLINE_SRC_DIR" >/dev/null
  log "Installing npm dependencies"
  npm install
  log "Running proto generation"
  npm run protos
  log "Running lint"
  npm run lint
  log "Building standalone bundle"
  node esbuild.mjs --standalone
  log "Packaging standalone artifacts"
  node scripts/package-standalone.mjs
  popd >/dev/null
}

bootstrap_task_workspace() {
  if [[ -n "$TASK_BOOTSTRAP_SCRIPT" ]]; then
    if [[ ! -x "$TASK_BOOTSTRAP_SCRIPT" ]]; then
      log "Bootstrap script $TASK_BOOTSTRAP_SCRIPT is not executable"
      exit 1
    fi
    log "Running task bootstrap script $TASK_BOOTSTRAP_SCRIPT"
    WORKSPACE_ROOT="$WORKSPACE_ROOT" "$TASK_BOOTSTRAP_SCRIPT" "$WORKSPACE_ROOT"
  else
    log "No TASK_BOOTSTRAP_SCRIPT provided; assuming workspace already prepared at $WORKSPACE_ROOT"
  fi
}

start_cline_core() {
  export NODE_OPTIONS
  export PROTOBUS_PORT HOSTBRIDGE_PORT
  export WORKSPACE_DIR=${WORKSPACE_DIR:-$WORKSPACE_ROOT}
  export DEV_WORKSPACE_FOLDER=${DEV_WORKSPACE_FOLDER:-$WORKSPACE_ROOT}
  export CLINE_DISABLE_BANNERS=true
  export CLINE_DISABLE_REMOTE_CONFIG=true
  export E2E_TEST=true
  export CLINE_ENVIRONMENT=local
  export PROJECT_ROOT="$CLINE_SRC_DIR"

  log "Launching Cline core in $WORKSPACE_DIR (using production server, no mock API)"
  
  # Use our custom server script that doesn't start ClineApiServerMock on port 7777
  # This allows multiple workers to run in parallel without port conflicts
  CUSTOM_SERVER="${ATROPOS_ROOT:-$ROOT_DIR/../..}/environments/cline_env/cline_dev/cline_core_server.ts"
  if [[ -f "$CUSTOM_SERVER" ]]; then
    log "Using custom server script: $CUSTOM_SERVER"
    pushd "$CLINE_SRC_DIR" >/dev/null
    npx tsx "$CUSTOM_SERVER"
    popd >/dev/null
  else
    log "Custom server not found, falling back to test server"
    pushd "$CLINE_SRC_DIR" >/dev/null
    npx tsx scripts/test-standalone-core-api-server.ts
    popd >/dev/null
  fi
}

main() {
  fetch_cline_repo
  build_cline
  bootstrap_task_workspace
  start_cline_core
}

main "$@"
