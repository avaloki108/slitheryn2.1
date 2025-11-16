# Quick Start Guide

## Installation

```bash
./install.sh
```

## Basic Usage

### Standard Analysis (use either)

```bash
# Regular Slither
slither contract.sol

# Slitheryn (same analysis + AI detector available)
slitheryn contract.sol
```

### AI-Enhanced Analysis

```bash
# Enable AI detector
slitheryn contract.sol --detect ai-analysis

# Multi-agent analysis
slitheryn contract.sol --multi-agent

# Quick scan with specific agents
slitheryn contract.sol --multi-agent \
  --agent-types vulnerability,exploit \
  --analysis-type quick
```

## Common Commands

```bash
# Standard Slither analysis
slither .

# Slitheryn with all AI features
slitheryn . --multi-agent --analysis-type comprehensive

# Focus on specific issues
slitheryn contract.sol --detect reentrancy-eth,ai-analysis

# Generate JSON output
slitheryn contract.sol --json output.json
```

## Verification

```bash
# Check installations
slither --version && slitheryn --version

# Verify they're separate
python3 -c "import slither; import slitheryn; print('slither:', slither.__file__); print('slitheryn:', slitheryn.__file__)"
```

## When to Use Which

- Use **`slither`** for:
  - Standard security audits
  - CI/CD integration
  - Fast analysis without AI
  
- Use **`slitheryn`** for:
  - Comprehensive security reviews
  - Complex DeFi protocols
  - When you need AI-powered insights
  - Multi-agent consensus analysis
