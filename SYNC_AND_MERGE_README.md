# SlitherYN2.1 - Sync & Merge Script Documentation

## Overview

The `sync_and_merge.sh` script automates the process of syncing your SlitherYN2.1 fork with the upstream Slither repository while preserving all your customizations. This eliminates the risk of losing your changes during updates.

## What This Script Does

1. **Stashes your customizations** - Creates a safe backup of all uncommitted changes
2. **Adds upstream remote** - Configures connection to the original Slither repository
3. **Fetches updates** - Downloads the latest changes from upstream
4. **Syncs with upstream** - Merges upstream updates into your repository
5. **Pushes to your repo** - Updates your remote repository with the synced code
6. **Restores customizations** - Applies your changes on top of the latest upstream code
7. **Commits changes** - Creates a commit with your merged customizations
8. **Detects conflicts** - Alerts you if there are merge conflicts that need manual resolution

## Prerequisites

- Git installed and configured
- SSH or HTTPS access to your GitHub repository
- Your fork must be based on the original Slither repository

## Usage

### Basic Usage

```bash
# From the repository root directory
./sync_and_merge.sh
```

### With Custom Repository URLs

```bash
./sync_and_merge.sh --your-repo origin --upstream-repo https://github.com/crytic/slither.git
```

### Available Options

- `--your-repo REMOTE` - Your remote repository name (default: `origin`)
- `--upstream-repo URL` - The upstream repository URL (default: `https://github.com/crytic/slither.git`)
- `--dry-run` - Show what would be done without making changes

## Output

The script creates a timestamped log file: `sync_log_YYYYMMDD_HHMMSS.txt`

Example output:
```
[2025-11-19 14:14:16] ========================================
[2025-11-19 14:14:16] SlitherYN2.1 Sync & Merge Script
[2025-11-19 14:14:16] ========================================
[✓] Customizations stashed
[✓] Upstream fetch complete
[✓] Upstream merge complete
[✓] Customizations restored
[✓] Sync and merge completed successfully!
```

## Stash Management

### View Your Stashes

```bash
git stash list
```

### Manually Restore a Stash (if needed)

```bash
# List stashes to find the one you want
git stash list

# Restore a specific stash
git stash pop stash@{0}

# Or apply without removing from stash list
git stash apply stash@{0}
```

### Current Stash

Your customizations have been stashed with the name:
```
Custom changes before sync - 20251119_141416
```

View it with:
```bash
git stash show "stash@{0}" -p
```

## Handling Merge Conflicts

If conflicts are detected:

1. The script will show which files have conflicts
2. Open each conflicted file and resolve the conflicts manually
3. Mark conflicts as resolved:
   ```bash
   git add <resolved-file>
   ```
4. Complete the merge:
   ```bash
   git commit -m "Resolve merge conflicts"
   ```

## Troubleshooting

### Script Failed - Need to Recover

```bash
# Check git status
git status

# View your stashed changes
git stash list

# Restore if needed
git stash pop
```

### Remote Tracking Issues

If you get errors about remote tracking:

```bash
# Verify remotes are set up correctly
git remote -v

# Update remote configuration
git remote set-url origin <your-fork-url>
```

## Automation

### Schedule with Cron (Linux/macOS)

```bash
# Run sync every week on Monday at 2 AM
0 2 * * 1 /path/to/slitheryn2.1/sync_and_merge.sh >> /var/log/slitheryn_sync.log 2>&1
```

### Run on Demand

```bash
# Manual execution from repo root
cd /home/dok/tools/W3-AUDIT/slitheryn2.1
./sync_and_merge.sh
```

## Best Practices

1. **Run during low-activity periods** - Schedule syncs when you're not actively developing
2. **Review conflicts** - Always check for merge conflicts after running the script
3. **Check logs** - Review the generated log files to ensure everything went smoothly
4. **Backup your repo** - Keep recent backups of your repository
5. **Test after sync** - Run your test suite after syncing to ensure compatibility

## Advanced: Customizing Sync Behavior

Edit the script to:

- Change default remote names
- Add custom commit messages
- Modify which branches to sync
- Add pre/post-sync hooks

## Support

For issues or improvements:
1. Check the log file for error messages
2. Verify git remotes: `git remote -v`
3. Review conflicts: `git status`
4. Check git history: `git log --oneline -10`

---

**Last Updated:** 2025-11-19
**Script Version:** 1.0
