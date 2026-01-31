# IDS Action Mapping & Evidence-Type Feature Partitioning Guide

## Overview

This guide documents the IDS-style action mapping and evidence-type based feature partitioning implemented in the VFL SHAP Multi-Class notebook. The improvements ensure that:

1. **Each party has different actions** based on attack type (not all parties have the same action)
2. **Features are partitioned by evidence type** (Volume/Rate, Packet Size, Timing/Direction) to promote different dominant parties
3. **Actions are attack-type specific** following IDS best practices

---

## Evidence-Type Feature Partitioning

### Strategy: Split by Evidence Type (Sensor-Based)

Instead of random or round-robin partitioning, features are now grouped by **evidence type** - each party represents a different network sensor:

#### **Party 1: Volume/Rate Evidence (DoS/DDoS Detection)**
- **Features:** Flow duration, total packet/byte counts, flow rates
- **Examples:**
  - `Flow Duration`
  - `Total Fwd Packets`, `Total Backward Packets`
  - `Total Length of Fwd Packets`, `Total Length of Bwd Packets`
  - `Flow Bytes/s`, `Flow Packets/s`
- **Why:** DoS/DDoS attacks show high volume/rate patterns
- **Expected Dominance:** DDOS, DOS classes

#### **Party 2: Packet Size Distribution (Scan/Web Attack Detection)**
- **Features:** Packet length statistics, size distributions, header lengths
- **Examples:**
  - `Fwd Packet Length Mean/Std/Max/Min`
  - `Bwd Packet Length Mean/Std/Max/Min`
  - `Packet Length Mean/Std/Variance`
  - `Avg Packet Size`, `Header Length`
- **Why:** Scans and web attacks show distinctive packet size patterns
- **Expected Dominance:** PORTSCAN, WEBATTACK classes

#### **Party 3: Timing/Directionality (Brute Force/Scan Detection)**
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

### Party-Specific Actions

Each party has **different actions** for different attack types:

#### **Party 1 (Volume/Rate Sensor)**
- **Primary Detection:** DDOS, DOS
  - **Action:** `rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers`
- **Secondary Detection:** Other attacks
  - **Action:** `monitor volume/rate patterns and alert`

#### **Party 2 (Packet Size Sensor)**
- **Primary Detection:** PORTSCAN, WEBATTACK
  - **PORTSCAN:** `scan detection: block scanner IP, tarpitting, tighten firewall rules`
  - **WEBATTACK:** `WAF rules, block patterns, patching, isolate vulnerable service`
- **Secondary Detection:** Other attacks
  - **Action:** `monitor packet size patterns and alert`

#### **Party 3 (Timing/Direction Sensor)**
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
3. **Priority 3:** Fallback rules (e.g., protocol/flag features → Party 3)

### Action Generation

The `get_party_actions_for_attack()` function:
- Returns **primary action** if party is primary detector for attack type
- Returns **secondary action** (monitoring) if party is secondary detector
- Uses `party_action_mapping` dictionary for fast lookup

### SHAP Integration

In Cell 12 (Agent-Level Mitigation Summary):
- Each party's action is **attack-type specific**
- Dominant party shows primary action for that attack
- Other parties show secondary actions (monitoring)
- CSV output includes attack-specific actions per party

---

## Expected Results

### Feature Distribution

After partitioning, you should see:
- **Party 1:** ~25-35 features (volume/rate)
- **Party 2:** ~20-30 features (packet size)
- **Party 3:** ~30-40 features (timing/direction/protocol)

### SHAP Dominance Patterns

Expected dominant parties per attack type:
- **DDOS/DOS:** Party 1 (Volume/Rate) should dominate
- **PORTSCAN:** Party 2 (Packet Size) or Party 3 (Timing) should dominate
- **WEBATTACK:** Party 2 (Packet Size) should dominate
- **SSHPATATOR/FTPPATATOR:** Party 3 (Timing/Direction) should dominate

### Action Diversity

**Before:** All parties had same generic action
```
Party 1: "configure packet-level security policies"
Party 2: "configure packet-level security policies"
Party 3: "configure packet-level security policies"
```

**After:** Each party has attack-specific actions
```
DDOS detected:
  Party 1: "rate-limit, SYN cookies, WAF rules, drop bursts, auto-scale, block top talkers"
  Party 2: "monitor packet size patterns and alert"
  Party 3: "monitor timing/direction patterns and alert"

SSHPATATOR detected:
  Party 1: "monitor volume/rate patterns and alert"
  Party 2: "monitor packet size patterns and alert"
  Party 3: "brute-force controls: fail2ban-style blocking, lockout, MFA, geo/IP reputation"
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

## Usage in Notebook

### Cell 2: Feature Partitioning
- Features are categorized by evidence type
- Parties are assigned based on evidence type
- Party names/domains reflect evidence type

### Cell 12: Mitigation Summary
- Uses `party_action_mapping` for attack-specific actions
- Shows dominant party action + other party actions
- CSV output includes attack-specific actions

### Output Files
- `vfl_shap_agent_mitigation_summary.csv`: Contains attack-specific actions per party
- `vfl_shap_{class}_summary.csv`: Per-class SHAP summaries with actions

---

## Troubleshooting

### Issue: All parties still have same actions

**Check:**
- Verify `party_action_mapping` is created in Cell 2
- Check that `get_party_actions_for_attack()` is called correctly
- Ensure attack type names match (uppercase)

### Issue: Wrong party dominates for attack type

**Check:**
- Verify feature categorization (run Cell 2 and check feature distribution)
- Check if features are correctly assigned to evidence types
- May need to refine `categorize_feature_by_evidence()` patterns

### Issue: Features not distributed evenly

**Solution:**
- Current distribution is intentional (evidence-based, not balanced)
- If needed, can add balancing logic after evidence categorization
- Or use Strategy 3 (balance by predictive power) from requirements

---

## Future Improvements

1. **Dynamic Action Selection:** Based on SHAP contribution threshold
2. **Multi-Party Actions:** Combine actions from multiple parties if all contribute significantly
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
