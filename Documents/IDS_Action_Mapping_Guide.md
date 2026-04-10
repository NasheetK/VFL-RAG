# IDS Action Mapping & Evidence-Type Feature Partitioning Guide

## Overview

This guide documents the IDS-style action mapping and evidence-type-based feature partitioning used with **`Train.ipynb`** (VFL + SHAP multi-class training) and supporting helpers in **`utils/vfl_utils.py`**. Runtime feature membership for each tier is defined in **`RAG_docs/agentic_features.json`** (schema version 1.1: `RAN`, `Edge`, `Core` agents with `logged_features` and `action_capabilities`). The improvements ensure that:

1. **Each party has different actions** based on attack type (not all parties have the same action)
2. **Features are partitioned by evidence type** (Volume/Rate, Packet Size, Timing/Direction) to promote different dominant parties
3. **Actions are attack-type specific** following IDS best practices

---

## Evidence-Type Feature Partitioning

### Strategy: Split by Evidence Type (Sensor-Based)

Instead of random or round-robin partitioning, features are now grouped by **evidence type** - each party represents a different network sensor:

#### **Agent 1: Volume/Rate Evidence (DoS/DDoS Detection)**
- **Features:** Flow duration, total packet/byte counts, flow rates
- **Examples:**
  - `Flow Duration`
  - `Total Fwd Packets`, `Total Backward Packets`
  - `Total Length of Fwd Packets`, `Total Length of Bwd Packets`
  - `Flow Bytes/s`, `Flow Packets/s`
- **Why:** DoS/DDoS attacks show high volume/rate patterns
- **Expected Dominance:** DDOS, DOS classes

#### **Agent 2: Packet Size Distribution (Scan/Web Attack Detection)**
- **Features:** Packet length statistics, size distributions, header lengths
- **Examples:**
  - `Fwd Packet Length Mean/Std/Max/Min`
  - `Bwd Packet Length Mean/Std/Max/Min`
  - `Packet Length Mean/Std/Variance`
  - `Avg Packet Size`, `Header Length`
- **Why:** Scans and web attacks show distinctive packet size patterns
- **Expected Dominance:** PORTSCAN, WEBATTACK classes

#### **Agent 3: Timing/Directionality (Brute Force/Scan Detection)**
- **Features:** IAT, timing intervals, active/idle, directionality ratios, burst patterns
- **Examples:**
  - `Fwd IAT Mean/Std/Max/Min`
  - `Bwd IAT Mean/Std/Max/Min`
  - `Active Mean/Std`, `Idle Mean/Std`
  - `Down/Up Ratio`, `Src2Dst`, `Dst2Src`
  - `SYN/ACK/FIN/RST Flag Count`
- **Why:** Brute force and scanning show timing/directionality patterns
- **Expected Dominance:** SSHPATATOR, FTPPATATOR, PORTSCAN classes

---

## IDS-Style Attack-Type Action Mapping

### Action Dictionary

Each attack type has a specific IDS-style action:

```python
ATTACK_ACTIONS = {
    'BENIGN': "✅ No action, log only",
    'DDOS': "rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers",
    'DOS': "rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers",
    'SSHPATATOR': "brute-force controls: fail2ban-style blocking, lockout, MFA, geo/IP reputation",
    'FTPPATATOR': "brute-force controls: fail2ban-style blocking, lockout, MFA, geo/IP reputation",
    'PORTSCAN': "scan detection: block scanner IP, tarpitting, tighten firewall rules",
    'WEBATTACK': "WAF rules, block patterns, patching, isolate vulnerable service",
    'BOT': "bot detection: block bot IPs, implement CAPTCHA, rate limiting",
    'OTHERS': "safe response: alert + collect evidence + temporary throttling (not hard block)"
}
```

### Agent-Specific Actions

Each party has **different actions** for different attack types:

#### **Agent 1 (Volume/Rate Sensor)**
- **Primary Detection:** DDOS, DOS
  - **Action:** `rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers`
- **Secondary Detection:** Other attacks
  - **Action:** `monitor volume/rate patterns and alert`

#### **Agent 2 (Packet Size Sensor)**
- **Primary Detection:** PORTSCAN, WEBATTACK
  - **PORTSCAN:** `scan detection: block scanner IP, tarpitting, tighten firewall rules`
  - **WEBATTACK:** `WAF rules, block patterns, patching, isolate vulnerable service`
- **Secondary Detection:** Other attacks
  - **Action:** `monitor packet size patterns and alert`

#### **Agent 3 (Timing/Direction Sensor)**
- **Primary Detection:** SSHPATATOR, FTPPATATOR, PORTSCAN
  - **SSHPATATOR/FTPPATATOR:** `brute-force controls: fail2ban-style blocking, lockout, MFA, geo/IP reputation`
  - **PORTSCAN:** `scan detection: block scanner IP, tarpitting, tighten firewall rules`
- **Secondary Detection:** Other attacks
  - **Action:** `monitor timing/direction patterns and alert`

---

## Implementation Details

### Feature Categorization Function

The `categorize_feature_by_evidence()` function uses priority-based matching:

1. **Priority 1:** Exact feature name matches (e.g., `flow_duration`, `packet_length_mean`)
2. **Priority 2:** Pattern matching (e.g., features with "total" + "packet" but not "length")
3. **Priority 3:** Fallback rules (e.g., protocol/flag features → Agent 3)

### Action Generation

The notebook uses **`get_agent_actions_for_attack()`** (and the legacy alias **`get_party_actions_for_attack()`** in `utils/vfl_utils.py`) to map **attack type + evidence type** to a primary mitigation string or a monitoring string for non-primary tiers. **`Train.ipynb`** also builds **`agent_action_mapping`** (per-agent, per-attack suggested strings) for summaries and exports.

### SHAP Integration

In **`Train.ipynb`**, search for the section **`# 12. Agent-Level Mitigation Summary`**:
- Each party's action is **attack-type specific**
- Dominant party shows primary action for that attack
- Other parties show secondary actions (monitoring)
- CSV output includes attack-specific actions per party

---

## Expected Results

### Feature distribution

From **`RAG_docs/agentic_features.json`** (current project):
- **RAN:** 23 logged features  
- **Edge:** 32 logged features  
- **Core:** 42 logged features  

Some flow columns appear in more than one agent list; the **union** across agents is **88** unique feature names, matching the 88-dimensional model input after preprocessing.

### SHAP dominance patterns

Expected dominant parties per attack type (Agent 1 = volume/rate tier, 2 = packet/size tier, 3 = timing/direction tier in the evidence-based story):
- **DDOS/DOS:** Agent 1 should often dominate
- **PORTSCAN:** Agent 2 or Agent 3 should dominate depending on flow pattern
- **WEBATTACK:** Agent 2 should dominate
- **SSHPATATOR/FTPPATATOR:** Agent 3 should dominate

### Action Diversity

**Before:** All parties had same generic action
```
Agent 1: "configure packet-level security policies"
Agent 2: "configure packet-level security policies"
Agent 3: "configure packet-level security policies"
```

**After:** Each party has attack-specific actions
```
DDOS detected:
  Agent 1: "rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers"
  Agent 2: "monitor packet size patterns and alert"
  Agent 3: "monitor timing/direction patterns and alert"

SSHPATATOR detected:
  Agent 1: "monitor volume/rate patterns and alert"
  Agent 2: "monitor packet size patterns and alert"
  Agent 3: "brute-force controls: fail2ban-style blocking, lockout, MFA, geo/IP reputation"
```

---

## Benefits

### 1. **Different Dominant Parties**
- Evidence-type partitioning ensures different parties dominate for different attacks
- Prevents single party from always winning (imbalanced splitting issue)

### 2. **Attack-Specific Actions**
- Each party provides appropriate action for each attack type
- Primary detector gets full action, secondary detectors monitor

### 3. **IDS Best Practices**
- Actions follow real-world IDS response strategies
- Different attack types get different mitigation approaches

### 4. **Better Interpretability**
- SHAP values show which sensor (party) detected the attack
- Actions align with sensor capabilities

---

## Usage in `Train.ipynb`

### Early cells: load, partition, and mappings
- Dataset load, stratified split, and agent feature lists from **`agentic_features.json`**
- **`agent_action_mapping`** populated via **`get_agent_actions_for_attack`**

### Mitigation summary (section ~12)
- Uses **`agent_action_mapping`** for attack-specific suggested actions in summaries
- Shows dominant party action + other party actions
- CSV output includes attack-specific actions per party

### Output Files
- `vfl_shap_agent_mitigation_summary.csv`: Contains attack-specific actions per party
- `vfl_shap_{class}_summary.csv`: Per-class SHAP summaries with actions

---

## Troubleshooting

### Issue: All parties still have same actions

**Check:**
- Verify **`agent_action_mapping`** is built after agent definitions are loaded
- Check that **`get_agent_actions_for_attack()`** (or **`get_party_actions_for_attack()`**) is used consistently
- Ensure attack type names match (uppercase)

### Issue: Wrong party dominates for attack type

**Check:**
- Verify feature categorization (re-run the partitioning / evidence logic in **`Train.ipynb`** and inspect distributions)
- Check if features are correctly assigned to evidence types
- May need to refine `categorize_feature_by_evidence()` patterns in **`utils/vfl_utils.py`**

### Issue: Features not distributed evenly

**Solution:**
- Current distribution is intentional (evidence-based, not balanced)
- If needed, can add balancing logic after evidence categorization
- Or use Strategy 3 (balance by predictive power) from requirements

---

## Future Improvements

1. **Dynamic Action Selection:** Based on SHAP contribution threshold
2. **Multi-Agent Actions:** Combine actions from multiple parties if all contribute significantly
3. **Action Priority:** Rank actions by severity/impact
4. **Feature Rebalancing:** Option to balance parties by feature importance scores

---

## Summary

✅ **Evidence-type partitioning** ensures different parties for different attacks
✅ **Attack-specific actions** provide IDS-style responses
✅ **Primary/secondary detection** distinguishes party roles
✅ **SHAP integration** shows which sensor detected which attack
✅ **Action diversity** prevents all parties from having same action

The system now provides **actionable, attack-specific mitigation recommendations** based on which network sensor (party) detected the attack.
