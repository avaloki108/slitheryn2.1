// Test contract for multi-agent analysis
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VulnerableContract {
    mapping(address => uint256) public balances;
    address public owner;
    uint256 public totalSupply;
    
    constructor() {
        owner = msg.sender;
        totalSupply = 1000000 * 10**18;
        balances[owner] = totalSupply;
    }
    
    // Reentrancy vulnerability
    function withdraw(uint256 amount) external {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // External call before state update (reentrancy vulnerability)
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount; // State update after external call
    }
    
    // Access control issue
    function mint(address to, uint256 amount) external {
        // Missing onlyOwner modifier
        balances[to] += amount;
        totalSupply += amount;
    }
    
    // Time manipulation vulnerability
    function timeBasedReward() external view returns (uint256) {
        if (block.timestamp % 2 == 0) {
            return 100;
        }
        return 50;
    }
    
    // tx.origin vulnerability
    function transferOwnership(address newOwner) external {
        require(tx.origin == owner, "Only owner"); // Should use msg.sender
        owner = newOwner;
    }
    
    // Integer overflow potential (if not using SafeMath)
    function addBalance(address user, uint256 amount) external {
        balances[user] += amount; // Potential overflow
    }
    
    receive() external payable {
        balances[msg.sender] += msg.value;
    }
}