# Slitheryn 2.1 - AI-Enhanced Smart Contract Security Analyzer

Slitheryn 2.1 is an AI-enhanced fork of Slither that integrates SmartLLM-OG and multi-agent analysis for comprehensive smart contract security analysis.

## Dual Command Setup

This repository is set up to work alongside the standard Slither installation:

- **`slither`** - Standard Slither static analyzer (from `/home/dok/tools/slither`)
- **`slitheryn`** - AI-enhanced analyzer with multi-agent capabilities (this repository)

## Installation

```bash
# Run the installation script
./install.sh
```

This will:
1. Install the regular Slither from `../slither` as the `slither` command
2. Install Slitheryn from this directory as the `slitheryn` command

### Manual Installation

```bash
# Install regular Slither
cd ../slither
pip install -e .

# Install Slitheryn
cd ../slitheryn2.1
pip install -e .
```

## Verify Installation

```bash
# Check both commands are available
slither --version
slitheryn --version

# Check they point to correct locations
which slither    # Should show ~/.local/bin/slither
which slitheryn  # Should show ~/.local/bin/slitheryn
```

## Usage

### Standard Analysis

Use `slither` for regular static analysis:

```bash
slither contract.sol
slither . --detect reentrancy-eth
```

Use `slitheryn` for the same static analysis with AI detector available:

```bash
slitheryn contract.sol
slitheryn . --detect reentrancy-eth
```

### AI-Enhanced Analysis

Slitheryn adds powerful AI-enhanced features:

```bash
# Basic AI analysis (requires Ollama)
slitheryn contract.sol --detect ai-analysis

# Multi-agent analysis
slitheryn contract.sol --multi-agent

# Comprehensive multi-agent analysis with all agents
slitheryn contract.sol --multi-agent \
  --agent-types vulnerability,exploit,fix,economic,governance \
  --analysis-type comprehensive

# Quick multi-agent scan
slitheryn contract.sol --multi-agent --analysis-type quick

# Custom consensus threshold
slitheryn contract.sol --multi-agent --consensus-threshold 0.8
```

## AI Features

### Multi-Agent Analysis

Slitheryn includes five specialized AI agents:

1. **Vulnerability Agent** - Identifies security vulnerabilities
2. **Exploit Agent** - Analyzes potential exploit scenarios
3. **Fix Agent** - Suggests remediation strategies
4. **Economic Agent** - Evaluates economic/financial risks
5. **Governance Agent** - Analyzes governance mechanisms

### CLI Options

```bash
--multi-agent                Enable multi-agent AI analysis
--agent-types TYPES          Comma-separated list of agents to use
--analysis-type TYPE         quick|comprehensive|specialized
--consensus-threshold VAL    Consensus threshold (0.0-1.0, default: 0.7)
--no-parallel-agents         Disable parallel agent execution
```

## Prerequisites for AI Features

### Ollama Setup

AI features require Ollama to be running:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull qwen2.5-coder:32b
ollama pull gemma2:27b
ollama pull mistral:7b
ollama pull llama3.1:70b

# Start Ollama server (if not already running)
ollama serve
```

### AI Configuration

Create `~/.slitheryn/config.yaml`:

```yaml
ai:
  enabled: true
  ollama_url: "http://localhost:11434"
  
  models:
    primary:
      - "qwen2.5-coder:32b"
      - "gemma2:27b"
    fallback:
      - "mistral:7b"
      - "llama3.1:70b"
  
  multi_agent:
    enabled: true
    agent_types:
      - vulnerability
      - exploit
      - fix
      - economic
      - governance
    consensus_threshold: 0.7
    parallel_analysis: true
    max_retries: 3
```

## Architecture

### Directory Structure

```
slitheryn2.1/
├── slitheryn/              # Main package (renamed from 'slither')
│   ├── ai/                 # AI integration module
│   │   ├── config.py       # AI configuration
│   │   └── ollama_client.py # Ollama API client
│   ├── detectors/
│   │   └── ai/             # AI-powered detectors
│   │       └── ai_enhanced_analysis.py
│   └── ...
├── integrations/           # SmartLLM-OG integration
├── test_contract.sol       # Test contract
├── test_vulnerable_contract.sol
├── install.sh              # Installation script
└── README.md               # This file
```

## Development

### Running Tests

```bash
# Test regular analysis
slitheryn test_contract.sol

# Test AI analysis (requires Ollama)
slitheryn test_vulnerable_contract.sol --detect ai-analysis

# Test multi-agent
slitheryn test_vulnerable_contract.sol --multi-agent --analysis-type quick
```

### Adding Custom Agents

See `slitheryn/ai/` for examples of how to add custom AI agents.

## Differences from Standard Slither

1. **Package Name**: `slitheryn-analyzer` vs `slither-analyzer`
2. **Command**: `slitheryn` vs `slither`
3. **Module**: `import slitheryn` vs `import slither`
4. **AI Features**: Multi-agent analysis, AI-enhanced detectors
5. **Branding**: References updated to Slitheryn throughout

## Migration from Slitheryn 1.0

If you're migrating customizations from the original Slitheryn:

```bash
# The migration script is already in this repo
python migrate_customizations.py
```

## Troubleshooting

### Both commands point to the same installation

```bash
# Uninstall both
pip uninstall slither-analyzer slitheryn-analyzer

# Reinstall using the script
./install.sh
```

### AI features not working

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Check models are available
ollama list

# Enable debug logging
slitheryn contract.sol --multi-agent --debug
```

### Import errors

If you see import errors, make sure the package was renamed correctly:

```bash
# Check the installation
python3 -c "import slitheryn; print(slitheryn.__file__)"

# Should point to this repository
```

## Contributing

Contributions welcome! Please:

1. Keep regular `slither` functionality intact
2. Add AI features as optional enhancements
3. Follow existing code style
4. Add tests for new features

## License

AGPL-3.0 (same as Slither)

## Credits

- Based on [Slither](https://github.com/crytic/slither) by Trail of Bits
- AI enhancements by [avaloki108](https://github.com/avaloki108)
- SmartLLM-OG integration

## Links

- [Slither Documentation](https://github.com/crytic/slither/wiki)
- [Ollama Documentation](https://ollama.ai/docs)
- [SmartLLM-OG](https://github.com/jasidok/SmartLLM-OG)
