# Action metrics vs `attack_options.json` (per sample)

Mitigation coverage = distinct allowed actions found in **`all_actions` ∪ primary_actions[].action ∪ supporting_actions[].action** / count of allowed actions for `predicted_label`. Avg. actions = total slots: len(all_actions) + #primary + #supporting. **Multi-layer** = ≥2 agents where a **primary/supporting** row has `network_tier` → RAN|Edge|Core and that agent's `action_capabilities` (in `agentic_features.json`) contains the **same** action. `all_actions` only gets an audit row (which agents list the action); it does not add layers (no tier on those entries). **Agentic tier–action alignment** = %% of primary/supporting rows whose action is listed under that row's agent in `action_capabilities` (100%% = all such rows match).

## Sample 0 — **BENIGN** — `rerank_comparison_sample_0_20260411_053200_573085.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| Matched actions | log incident, monitor traffic | log incident, monitor traffic | log incident, monitor traffic |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (2/4) | 50% (2/4) | 50% (2/4) |
| Agentic tier–action alignment | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| Multi-layer strategies | No (RAN only) | No (RAN only) | No (RAN only) |

## Sample 1 — **BOT** — `rerank_comparison_sample_1_20260411_053250_993527.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 33.33% (2/6) | 50% (3/6) | 33.33% (2/6) |
| Matched actions | captcha challenge, limit rate | captcha challenge, limit rate, apply WAF | captcha challenge, limit rate |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (2/4) | 50% (3/6) | 50% (2/4) |
| Agentic tier–action alignment | 100% (2/2) | 66.67% (2/3) | 100% (2/2) |
| Multi-layer strategies | No (Edge only) | No (Edge only) | No (Edge only) |

## Sample 2 — **DDOS** — `rerank_comparison_sample_2_20260411_053336_937148.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 33.33% (2/6) | 33.33% (2/6) | 33.33% (2/6) |
| Matched actions | limit rate, enable scrubbing | limit rate, enable scrubbing | limit rate, enable scrubbing |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (2/4) | 50% (2/4) | 50% (2/4) |
| Agentic tier–action alignment | 50% (1/2) | 50% (1/2) | 50% (1/2) |
| Multi-layer strategies | No (RAN only) | No (RAN only) | No (RAN only) |

## Sample 3 — **DOS** — `rerank_comparison_sample_3_20260411_053426_734082.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 50% (3/6) | 50% (3/6) | 50% (3/6) |
| Matched actions | enable syncookies, limit rate, block IP | enable syncookies, limit rate, block IP | enable syncookies, limit rate, block IP |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (3/6) | 50% (3/6) | 50% (3/6) |
| Agentic tier–action alignment | 66.67% (2/3) | 66.67% (2/3) | 66.67% (2/3) |
| Multi-layer strategies | No (RAN only) | No (RAN only) | No (RAN only) |

## Sample 4 — **FTPPATATOR** — `rerank_comparison_sample_4_20260411_053517_026234.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 100% (6/6) | 50% (3/6) | 50% (3/6) |
| Matched actions | lock account, enforce MFA, throttle credentials, fail2ban block, block IP, update ACL | lock account, enforce MFA, throttle credentials | lock account, enforce MFA, throttle credentials |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (6/12) | 50% (3/6) | 50% (3/6) |
| Agentic tier–action alignment | 100% (6/6) | 100% (3/3) | 100% (3/3) |
| Multi-layer strategies | No (Edge only) | No (Edge only) | No (Edge only) |

## Sample 5 — **OTHERS** — `rerank_comparison_sample_5_20260411_053626_191568.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 50% (3/6) | 33.33% (2/6) | 33.33% (2/6) |
| Matched actions | limit rate, update ACL, block IP | limit rate, update ACL | limit rate, update ACL |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (3/6) | 50% (2/4) | 50% (2/4) |
| Agentic tier–action alignment | 100% (3/3) | 100% (2/2) | 100% (2/2) |
| Multi-layer strategies | No (RAN only) | No (RAN only) | No (RAN only) |

## Sample 6 — **PORTSCAN** — `rerank_comparison_sample_6_20260411_053716_338633.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 33.33% (2/6) | 33.33% (2/6) | 33.33% (2/6) |
| Matched actions | block IP, limit rate | block IP, limit rate | block IP, limit rate |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (2/4) | 50% (2/4) | 50% (2/4) |
| Agentic tier–action alignment | 100% (2/2) | 100% (2/2) | 100% (2/2) |
| Multi-layer strategies | No (RAN only) | No (RAN only) | No (RAN only) |

## Sample 7 — **SSHPATATOR** — `rerank_comparison_sample_7_20260411_053839_148503.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 83.33% (5/6) | 50% (3/6) | 83.33% (5/6) |
| Matched actions | lock account, enforce MFA, throttle credentials, fail2ban block, block IP | lock account, enforce MFA, throttle credentials | lock account, enforce MFA, throttle credentials, fail2ban block, block IP |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (5/10) | 50% (3/6) | 50% (5/10) |
| Agentic tier–action alignment | 100% (5/5) | 100% (3/3) | 100% (5/5) |
| Multi-layer strategies | No (Edge only) | No (Edge only) | No (Edge only) |

## Sample 8 — **WEBATTACK** — `rerank_comparison_sample_8_20260411_053928_711793.json`

| Metric | MMR | CrossEncoder | ColBERT |
| --- | --- | --- | --- |
| Mitigation coverage | 50% (3/6) | 33.33% (2/6) | 33.33% (2/6) |
| Matched actions | apply WAF, block IP, limit rate | apply WAF, block IP | apply WAF, block IP |
| All known mitigations | YES | YES | YES |
| Action scenario (unique/total) | 50% (3/6) | 50% (2/4) | 50% (2/4) |
| Agentic tier–action alignment | 100% (3/3) | 100% (2/2) | 100% (2/2) |
| Multi-layer strategies | No (Edge only) | No (Edge only) | No (Edge only) |
