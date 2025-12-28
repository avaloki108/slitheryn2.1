#!/usr/bin/env bash

### Test slitheryn-prop

cd examples/slither-prop || exit 1
slitheryn-prop . --contract ERC20Buggy
if [ ! -f contracts/crytic/TestERC20BuggyTransferable.sol ]; then
    echo "slitheryn-prop failed"
    return 1
fi
