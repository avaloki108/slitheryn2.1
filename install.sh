#!/bin/bash
# Installation script for both Slither and Slitheryn

set -e

echo "🚀 Installing Slither and Slitheryn side-by-side"
echo "================================================"

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "📍 Directories:"
echo "   Slither:   $PARENT_DIR/slither"
echo "   Slitheryn: $SCRIPT_DIR"

# Check if slither directory exists
if [ ! -d "$PARENT_DIR/slither" ]; then
    echo ""
    echo "❌ Error: Slither directory not found at $PARENT_DIR/slither"
    echo "   Please ensure the regular Slither is installed at ../slither"
    exit 1
fi

# Uninstall any existing installations to avoid conflicts
echo ""
echo "🧹 Cleaning up existing installations..."
pip uninstall -y slither-analyzer slitheryn-analyzer 2>/dev/null || true

# Install regular Slither first
echo ""
echo "📦 Installing Slither (regular static analyzer)..."
cd "$PARENT_DIR/slither"
pip install -e .

# Verify slither installation
if ! command -v slither &> /dev/null; then
    echo "❌ Error: Slither installation failed"
    exit 1
fi

echo "✅ Slither installed successfully"
slither --version

# Install Slitheryn
echo ""
echo "📦 Installing Slitheryn (AI-enhanced analyzer)..."
cd "$SCRIPT_DIR"
pip install -e .

# Verify slitheryn installation
if ! command -v slitheryn &> /dev/null; then
    echo "❌ Error: Slitheryn installation failed"
    exit 1
fi

echo "✅ Slitheryn installed successfully"
slitheryn --version

# Summary
echo ""
echo "================================================"
echo "✅ Installation Complete!"
echo ""
echo "Available commands:"
echo "  slither   - Regular static analyzer"
echo "  slitheryn - AI-enhanced analyzer"
echo ""
echo "Test the installations:"
echo "  slither --version"
echo "  slitheryn --version"
echo ""
echo "To use AI features with Slitheryn:"
echo "  slitheryn <contract.sol> --multi-agent"
echo "================================================"
