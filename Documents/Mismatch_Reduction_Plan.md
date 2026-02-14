# Plan: Reduce Detection-Mitigation Mismatch

## Problem Analysis

### Current Mismatches

| Attack Type | Dominant Agent | Current Action | Primary Action Agent | Issue |
|------------|----------------|----------------|---------------------|-------|
| **DDOS** | Agent 3 (44.84%) | "monitor timing/direction" | Agent 1 | Agent 3 detects but Agent 1 should mitigate |
| **SSHPATATOR** | Agent 1 (47.28%) | "monitor volume/rate" | Agent 3 | Agent 1 detects but Agent 3 should mitigate |
| **WEBATTACK** | Agent 1 (71.68%) | "monitor volume/rate" | Agent 2 | Agent 1 detects but Agent 2 should mitigate |

### Root Cause

The current system assigns actions based on:
- **Evidence type** (which party's features match the attack pattern)
- **Primary attack mapping** (predefined which party handles which attack)

But SHAP shows **actual detection dominance**, which may differ from predefined mappings.

---

## Solution Strategy

### Approach 1: Smart Reassignment (Recommended)

**Logic:**
1. If dominant party has primary action → **Keep it** (no change needed)
2. If dominant party doesn't have primary action BUT has capabilities → **Reassign primary action to dominant party**
3. If dominant party can't mitigate → **Create coordinated action**

**Benefits:**
- Reduces mismatch from 3 cases to 0-1 cases
- Maintains detection-mitigation alignment
- Still allows coordination when needed

### Approach 2: Coordination Model

**Logic:**
- Dominant party: "monitor and coordinate with [Agent X] for: [action]"
- Action party: Execute the primary action

**Benefits:**
- Clear separation of detection vs. mitigation
- Realistic multi-party collaboration
- No reassignment needed

---

## Implementation Plan

### Step 1: Define Agent Capabilities

```python
PARTY_CAPABILITIES = {
    0: ['DDOS', 'DOS'],  # Agent 1: Volume/Rate -> DoS mitigation
    1: ['PORTSCAN', 'WEBATTACK'],  # Agent 2: Packet Size -> Scan/Web mitigation
    2: ['SSHPATATOR', 'FTPPATATOR', 'PORTSCAN']  # Agent 3: Timing -> Brute force mitigation
}
```

### Step 2: Create Improved Action Function

```python
def get_improved_action_for_party(party_idx, attack_type, is_dominant, 
                                  current_action, party_action_mapping):
    """
    Improved action assignment logic
    """
    attack_type_upper = attack_type.upper()
    primary_action = ATTACK_PRIMARY_ACTIONS.get(attack_type_upper)
    
    # Check if current action is primary
    is_primary = (current_action == primary_action)
    
    # Check if party has capabilities
    has_capabilities = attack_type_upper in PARTY_CAPABILITIES.get(party_idx, [])
    
    if is_dominant:
        if is_primary:
            return primary_action, "primary_dominant"  # Perfect match
        elif has_capabilities:
            return primary_action, "primary_reassigned"  # Reassign to dominant
        else:
            # Coordinate with action party
            return f"monitor and coordinate with [Agent] for: {primary_action}", "coordinated"
    else:
        return current_action, "standard"
```

### Step 3: Apply in Notebook

Replace this line in your mitigation summary generation:
```python
# OLD:
suggested_action = party_action_mapping[i].get(attack_type, party_actions[i])

# NEW:
improved_action, action_type = get_improved_action_for_party(
    party_idx=i,
    attack_type=attack_type,
    is_dominant=(i == top_party_idx),
    current_action=party_action_mapping[i].get(attack_type, party_actions[i]),
    party_action_mapping=party_action_mapping
)
suggested_action = improved_action
```

---

## Expected Results

### Before (Current Output)

```
DDOS (Attack Type: DDOS):
  Dominant Agent: Timing_Direction_Sensor_Agent3 (44.84%)
  Recommended Action: monitor timing/direction patterns and alert  ❌ Mismatch
  Other Agent Actions:
    Volume_Rate_Sensor_Agent1: rate-limit, SYN cookies...  ✅ Has action but not dominant
```

### After (Improved Output)

**Option A: Reassignment**
```
DDOS (Attack Type: DDOS):
  Dominant Agent: Timing_Direction_Sensor_Agent3 (44.84%)
  Recommended Action: monitor and coordinate with Agent 1 for: rate-limit, SYN cookies...  ✅ Coordinated
  Other Agent Actions:
    Volume_Rate_Sensor_Agent1: rate-limit, SYN cookies...  ✅ Executes
```

**Option B: Reassignment (if Agent 3 has capabilities)**
```
DDOS (Attack Type: DDOS):
  Dominant Agent: Timing_Direction_Sensor_Agent3 (44.84%)
  Recommended Action: rate-limit, SYN cookies...  ✅ Reassigned to dominant
  Other Agent Actions:
    Volume_Rate_Sensor_Agent1: monitor volume/rate patterns  ✅ Monitors
```

---

## Comparison Table

| Attack | Current Mismatch | After (Reassignment) | After (Coordination) |
|--------|------------------|---------------------|---------------------|
| DDOS | ❌ Agent 3 monitors, Agent 1 acts | ✅ Agent 3 coordinates, Agent 1 acts | ✅ Agent 3 coordinates, Agent 1 acts |
| SSHPATATOR | ❌ Agent 1 monitors, Agent 3 acts | ✅ Agent 1 coordinates, Agent 3 acts | ✅ Agent 1 coordinates, Agent 3 acts |
| WEBATTACK | ❌ Agent 1 monitors, Agent 2 acts | ✅ Agent 1 coordinates, Agent 2 acts | ✅ Agent 1 coordinates, Agent 2 acts |

---

## Implementation Files

1. **`sample_src/Reduce_Mismatch_Solution.py`** - Complete solution with functions
2. **`sample_src/Improved_Action_Assignment.py`** - Alternative implementation
3. **This document** - Explanation and plan

---

## Next Steps

1. ✅ Review the solution files
2. ✅ Choose approach (Reassignment vs. Coordination)
3. ✅ Integrate into your notebook
4. ✅ Test with your data
5. ✅ Compare before/after results

---

## Notes

- **Reassignment** works best when parties have overlapping capabilities
- **Coordination** is more realistic for distributed systems
- You can combine both: reassign when possible, coordinate when needed
- The solution maintains backward compatibility (doesn't break existing code)
