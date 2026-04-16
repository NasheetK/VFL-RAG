# Explanations (per sample, per mode)

- **Mitigation coverage**: which allowed actions (from `attack_options.json`) were hit/missed.
- **All known mitigations**: YES iff all LLM action strings are a subset of the allowed list for that attack.
- **Action scenario (unique/total)**: diversity of action strings (unique distinct / total slots).
- **Agentic tier–action alignment**: for each primary/supporting row, whether the row's `network_tier` agent lists that `action` in `agentic_features.json` `action_capabilities`.
- **Multi-layer**: requires ≥2 distinct agents with at least one primary/supporting row where `agentic_match == true`.

## Sample 0 — **BENIGN** — `rerank_comparison_sample_0_20260411_053200_573085.json`

### MMR

- **Mitigation coverage**: 100% (2/2)
  - **matched**: log incident, monitor traffic
  - **missing**: —
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `log incident` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `monitor traffic` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### CrossEncoder

- **Mitigation coverage**: 100% (2/2)
  - **matched**: log incident, monitor traffic
  - **missing**: —
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `log incident` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `monitor traffic` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### ColBERT

- **Mitigation coverage**: 100% (2/2)
  - **matched**: log incident, monitor traffic
  - **missing**: —
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `log incident` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `monitor traffic` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)


## Sample 1 — **BOT** — `rerank_comparison_sample_1_20260411_053250_993527.json`

### MMR

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: captcha challenge, limit rate
  - **missing**: apply WAF, reputation filter, block IP, js challenge
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `captcha challenge` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `limit rate` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)

### CrossEncoder

- **Mitigation coverage**: 50% (3/6)
  - **matched**: captcha challenge, limit rate, apply WAF
  - **missing**: reputation filter, block IP, js challenge
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 66.67% (2/3)
  - **row checks (primary/supporting)**:
    - `captcha challenge` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `limit rate` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `apply WAF` @ `Core` (norm `Core`) → **NOT ALLOWED**; capability_agents=['Edge']
- **Multi-layer strategies**: No (Edge)

### ColBERT

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: captcha challenge, limit rate
  - **missing**: apply WAF, reputation filter, block IP, js challenge
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `captcha challenge` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `limit rate` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)


## Sample 2 — **DDOS** — `rerank_comparison_sample_2_20260411_053336_937148.json`

### MMR

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: limit rate, enable scrubbing
  - **missing**: blackhole route, update ACL, block IP, scale service
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 50% (1/2)
  - **row checks (primary/supporting)**:
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `enable scrubbing` @ `RAN` (norm `RAN`) → **NOT ALLOWED**; capability_agents=['Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### CrossEncoder

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: limit rate, enable scrubbing
  - **missing**: blackhole route, update ACL, block IP, scale service
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 50% (1/2)
  - **row checks (primary/supporting)**:
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `enable scrubbing` @ `RAN` (norm `RAN`) → **NOT ALLOWED**; capability_agents=['Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### ColBERT

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: limit rate, enable scrubbing
  - **missing**: blackhole route, update ACL, block IP, scale service
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 50% (1/2)
  - **row checks (primary/supporting)**:
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `enable scrubbing` @ `RAN` (norm `RAN`) → **NOT ALLOWED**; capability_agents=['Edge', 'Core']
- **Multi-layer strategies**: No (RAN)


## Sample 3 — **DOS** — `rerank_comparison_sample_3_20260411_053426_734082.json`

### MMR

- **Mitigation coverage**: 50% (3/6)
  - **matched**: enable syncookies, limit rate, block IP
  - **missing**: connection limit, update ACL, enable scrubbing
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 66.67% (2/3)
  - **row checks (primary/supporting)**:
    - `enable syncookies` @ `RAN` (norm `RAN`) → **NOT ALLOWED**; capability_agents=['Edge', 'Core']
    - `block IP` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### CrossEncoder

- **Mitigation coverage**: 50% (3/6)
  - **matched**: enable syncookies, limit rate, block IP
  - **missing**: connection limit, update ACL, enable scrubbing
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 66.67% (2/3)
  - **row checks (primary/supporting)**:
    - `enable syncookies` @ `RAN` (norm `RAN`) → **NOT ALLOWED**; capability_agents=['Edge', 'Core']
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `block IP` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### ColBERT

- **Mitigation coverage**: 50% (3/6)
  - **matched**: enable syncookies, limit rate, block IP
  - **missing**: connection limit, update ACL, enable scrubbing
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 66.67% (2/3)
  - **row checks (primary/supporting)**:
    - `enable syncookies` @ `RAN` (norm `RAN`) → **NOT ALLOWED**; capability_agents=['Edge', 'Core']
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `block IP` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)


## Sample 4 — **FTPPATATOR** — `rerank_comparison_sample_4_20260411_053517_026234.json`

### MMR

- **Mitigation coverage**: 100% (6/6)
  - **matched**: lock account, enforce MFA, throttle credentials, fail2ban block, block IP, update ACL
  - **missing**: —
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (6/12)
- **Agentic tier–action alignment**: 100% (6/6)
  - **row checks (primary/supporting)**:
    - `lock account` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `enforce MFA` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `throttle credentials` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `fail2ban block` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `block IP` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `update ACL` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)

### CrossEncoder

- **Mitigation coverage**: 50% (3/6)
  - **matched**: lock account, enforce MFA, throttle credentials
  - **missing**: fail2ban block, block IP, update ACL
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 100% (3/3)
  - **row checks (primary/supporting)**:
    - `lock account` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `enforce MFA` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `throttle credentials` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
- **Multi-layer strategies**: No (Edge)

### ColBERT

- **Mitigation coverage**: 50% (3/6)
  - **matched**: lock account, enforce MFA, throttle credentials
  - **missing**: fail2ban block, block IP, update ACL
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 100% (3/3)
  - **row checks (primary/supporting)**:
    - `lock account` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `enforce MFA` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `throttle credentials` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
- **Multi-layer strategies**: No (Edge)


## Sample 5 — **OTHERS** — `rerank_comparison_sample_5_20260411_053626_191568.json`

### MMR

- **Mitigation coverage**: 50% (3/6)
  - **matched**: limit rate, update ACL, block IP
  - **missing**: enable scrubbing, isolate service, log incident
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 100% (3/3)
  - **row checks (primary/supporting)**:
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `update ACL` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `block IP` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### CrossEncoder

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: limit rate, update ACL
  - **missing**: block IP, enable scrubbing, isolate service, log incident
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `update ACL` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### ColBERT

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: limit rate, update ACL
  - **missing**: block IP, enable scrubbing, isolate service, log incident
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `update ACL` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)


## Sample 6 — **PORTSCAN** — `rerank_comparison_sample_6_20260411_053716_338633.json`

### MMR

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: block IP, limit rate
  - **missing**: tarpit scan, update ACL, harden ports, scan threshold
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `block IP` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### CrossEncoder

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: block IP, limit rate
  - **missing**: tarpit scan, update ACL, harden ports, scan threshold
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `block IP` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)

### ColBERT

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: block IP, limit rate
  - **missing**: tarpit scan, update ACL, harden ports, scan threshold
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `block IP` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `limit rate` @ `RAN` (norm `RAN`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (RAN)


## Sample 7 — **SSHPATATOR** — `rerank_comparison_sample_7_20260411_053839_148503.json`

### MMR

- **Mitigation coverage**: 83.33% (5/6)
  - **matched**: lock account, enforce MFA, throttle credentials, fail2ban block, block IP
  - **missing**: update ACL
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (5/10)
- **Agentic tier–action alignment**: 100% (5/5)
  - **row checks (primary/supporting)**:
    - `lock account` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `enforce MFA` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `throttle credentials` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `fail2ban block` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `block IP` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)

### CrossEncoder

- **Mitigation coverage**: 50% (3/6)
  - **matched**: lock account, enforce MFA, throttle credentials
  - **missing**: fail2ban block, block IP, update ACL
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 100% (3/3)
  - **row checks (primary/supporting)**:
    - `lock account` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `enforce MFA` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `throttle credentials` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
- **Multi-layer strategies**: No (Edge)

### ColBERT

- **Mitigation coverage**: 83.33% (5/6)
  - **matched**: lock account, enforce MFA, throttle credentials, fail2ban block, block IP
  - **missing**: update ACL
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (5/10)
- **Agentic tier–action alignment**: 100% (5/5)
  - **row checks (primary/supporting)**:
    - `lock account` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `enforce MFA` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `throttle credentials` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `fail2ban block` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `block IP` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)


## Sample 8 — **WEBATTACK** — `rerank_comparison_sample_8_20260411_053928_711793.json`

### MMR

- **Mitigation coverage**: 50% (3/6)
  - **matched**: apply WAF, block IP, limit rate
  - **missing**: virtual patch, update ACL, isolate service
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (3/6)
- **Agentic tier–action alignment**: 100% (3/3)
  - **row checks (primary/supporting)**:
    - `apply WAF` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `block IP` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
    - `limit rate` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)

### CrossEncoder

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: apply WAF, block IP
  - **missing**: virtual patch, limit rate, update ACL, isolate service
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `apply WAF` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `block IP` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)

### ColBERT

- **Mitigation coverage**: 33.33% (2/6)
  - **matched**: apply WAF, block IP
  - **missing**: virtual patch, limit rate, update ACL, isolate service
- **All known mitigations**: YES
- **Action scenario (unique/total)**: 50% (2/4)
- **Agentic tier–action alignment**: 100% (2/2)
  - **row checks (primary/supporting)**:
    - `apply WAF` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['Edge']
    - `block IP` @ `Edge` (norm `Edge`) → **ALLOWED**; capability_agents=['RAN', 'Edge', 'Core']
- **Multi-layer strategies**: No (Edge)

