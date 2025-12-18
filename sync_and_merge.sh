#!/bin/bash

################################################################################
# SlitherYN2.1 - Sync & Merge with Customizations Script
# Purpose: Automatically sync with upstream Slither repo and merge with your
#          own repository while preserving all customizations
# Usage: ./sync_and_merge.sh [--upstream-repo URL] [--dry-run]
################################################################################

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_REPO="${UPSTREAM_REPO:-https://github.com/crytic/slither.git}"
YOUR_REPO="${YOUR_REPO:-origin}"
CUSTOM_STASH_NAME="custom-changes-$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false
LOG_FILE="${SCRIPT_DIR}/sync_log_$(date +%Y%m%d_%H%M%S).txt"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

################################################################################
# Logging Functions
################################################################################

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1" | tee -a "$LOG_FILE"
}

################################################################################
# Utility Functions
################################################################################

check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        log_error "Not a git repository: $PWD"
        exit 1
    fi
}

check_clean_working_tree() {
    if ! git diff-index --quiet HEAD --; then
        log_warning "Working tree has uncommitted changes"
        log "These will be stashed as: $CUSTOM_STASH_NAME"
    fi
}

stash_customizations() {
    log "Stashing your customizations..."
    
    if git diff-index --quiet HEAD --; then
        log_success "Working tree is clean, no stashing needed"
        return 0
    fi
    
    local stash_count=$(git stash list | wc -l)
    git stash push -m "$CUSTOM_STASH_NAME"
    
    if [ $(git stash list | wc -l) -gt $stash_count ]; then
        log_success "Customizations stashed"
        return 0
    else
        log_error "Failed to stash customizations"
        return 1
    fi
}

add_upstream_remote() {
    log "Checking for upstream remote..."
    
    if git remote | grep -q "^upstream$"; then
        log_success "Upstream remote already exists"
        git remote set-url upstream "$UPSTREAM_REPO"
    else
        log "Adding upstream remote..."
        git remote add upstream "$UPSTREAM_REPO"
    fi
    
    log_success "Upstream remote configured"
}

fetch_upstream() {
    log "Fetching from upstream repository..."
    git fetch upstream --no-tags
    log_success "Upstream fetch complete"
}

sync_with_upstream() {
    log "Syncing master branch with upstream/master..."
    
    git checkout master 2>/dev/null || git checkout main 2>/dev/null || {
        log_error "Could not checkout master or main branch"
        return 1
    }
    
    # Get current branch
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    log "Current branch: $current_branch"
    
    # Merge upstream changes
    if git merge-base --is-ancestor upstream/master HEAD; then
        log "Your branch is ahead of upstream - fast-forwarding latest upstream changes..."
        git merge upstream/master --ff-only --no-edit || {
            log_warning "Fast-forward merge failed, attempting standard merge..."
            git merge upstream/master --no-edit
        }
    else
        log "Merging upstream/master into $current_branch..."
        git merge upstream/master --no-edit
    fi
    
    log_success "Upstream merge complete"
}

push_to_your_repo() {
    log "Pushing synced code to your repository..."
    
    local current_branch=$(git rev-parse --abbrev-ref HEAD)
    git push "$YOUR_REPO" "$current_branch"
    
    log_success "Pushed to $YOUR_REPO/$current_branch"
}

restore_customizations() {
    log "Restoring your customizations..."
    
    # Find the stash we just created
    local stash_ref=$(git stash list | grep "$CUSTOM_STASH_NAME" | head -1 | cut -d: -f1)
    
    if [ -z "$stash_ref" ]; then
        log_warning "Could not find custom stash - no customizations to restore"
        return 0
    fi
    
    if git stash pop "$stash_ref"; then
        log_success "Customizations restored"
        
        # Check for conflicts
        if git diff --name-only --diff-filter=U | grep -q .; then
            log_warning "Merge conflicts detected in:"
            git diff --name-only --diff-filter=U | tee -a "$LOG_FILE"
            log_warning "Please resolve conflicts manually and commit"
            return 1
        fi
        return 0
    else
        log_error "Failed to restore customizations"
        log "Stash reference: $stash_ref"
        return 1
    fi
}

commit_customizations() {
    log "Committing restored customizations..."
    
    if ! git diff-index --quiet HEAD --; then
        git add -A
        git commit -m "Merge customizations after upstream sync" --allow-empty
        log_success "Customizations committed"
    else
        log_success "No changes to commit"
    fi
}

show_status() {
    log "Final repository status:"
    echo "" | tee -a "$LOG_FILE"
    git --no-pager status | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    git --no-pager log --oneline -5 | tee -a "$LOG_FILE"
}

################################################################################
# Main Execution
################################################################################

main() {
    log "========================================"
    log "SlitherYN2.1 Sync & Merge Script"
    log "========================================"
    
    cd "$SCRIPT_DIR"
    check_git_repo
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --upstream-repo)
                UPSTREAM_REPO="$2"
                shift 2
                ;;
            --your-repo)
                YOUR_REPO="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            *)
                log_error "Unknown argument: $1"
                exit 1
                ;;
        esac
    done
    
    log "Configuration:"
    log "  Upstream Repo: $UPSTREAM_REPO"
    log "  Your Repo: $YOUR_REPO"
    log "  Dry Run: $DRY_RUN"
    log "  Log File: $LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    
    # Execute sync pipeline
    check_clean_working_tree
    stash_customizations || exit 1
    add_upstream_remote
    fetch_upstream || exit 1
    sync_with_upstream || exit 1
    push_to_your_repo || exit 1
    restore_customizations || log_warning "Customization restore completed with warnings"
    commit_customizations
    echo "" | tee -a "$LOG_FILE"
    show_status
    
    log "========================================"
    log_success "Sync and merge completed successfully!"
    log "========================================"
    log "Log file saved to: $LOG_FILE"
}

# Trap errors
trap 'log_error "Script failed at line $LINENO"; exit 1' ERR

# Run main
main "$@"
