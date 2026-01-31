# How to Handle Multiple Attack Types in VFL_SHAP

## **Current Setup (Binary):**
- Label: 0 = Benign, 1 = Attack
- Model: Binary classification (sigmoid output)
- Analysis: Attack vs Benign only

## **Multi-Attack Type Setup:**
- Labels: 0 = Benign, 1 = DDoS, 2 = PortScan, 3 = BruteForce, 4 = Bot, etc.
- Model: Multi-class classification (softmax output)
- Analysis: Per attack type

---

## **Step 1: Modify Label Processing**

### **Current (Binary):**
```python
label_col = "label"   # binary 0/1
y = torch.tensor(df[label_col].values, dtype=torch.float32)
```

### **Multi-Class:**
```python
# Option 1: If you have attack type column
label_col = "Attack_Type"  # e.g., "BENIGN", "DDoS", "PortScan", etc.

# Map to numeric labels
attack_type_mapping = {
    "BENIGN": 0,
    "DDoS": 1,
    "PortScan": 2,
    "BruteForce": 3,
    "Bot": 4,
    # ... add more attack types
}

df['label_numeric'] = df[label_col].map(attack_type_mapping)
y = torch.tensor(df['label_numeric'].values, dtype=torch.long)  # long for multi-class

# Option 2: If label is already numeric (0,1,2,3...)
y = torch.tensor(df[label_col].values, dtype=torch.long)
num_classes = len(torch.unique(y))  # e.g., 5 classes
```

---

## **Step 2: Modify Model for Multi-Class**

### **Current (Binary):**
```python
class ActiveClassifier(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)  # 1 output (binary)
    
    def forward(self, h):
        return torch.sigmoid(self.fc(h))  # Sigmoid for binary
```

### **Multi-Class:**
```python
class ActiveClassifier(nn.Module):
    def __init__(self, embed_dim=32, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)  # 5 outputs (multi-class)
    
    def forward(self, h):
        return torch.softmax(self.fc(h), dim=1)  # Softmax for multi-class
```

### **Update VFLModel:**
```python
class VFLModel(nn.Module):
    def __init__(self, input_dims, embed_dim=32, num_classes=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoders = nn.ModuleList([LocalEncoder(dim, embed_dim) for dim in input_dims])
        self.classifier = ActiveClassifier(embed_dim, num_classes)  # Add num_classes
```

---

## **Step 3: Modify Loss Function**

### **Current (Binary):**
```python
criterion = nn.BCELoss()  # Binary Cross-Entropy
```

### **Multi-Class:**
```python
criterion = nn.CrossEntropyLoss()  # Multi-class Cross-Entropy
```

### **Update Training:**
```python
def train_vfl(model, x_parts, y, epochs=100, lr=1e-3):
    criterion = nn.CrossEntropyLoss()  # Changed
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        y_hat = model(x_parts)  # [batch, num_classes]
        loss = criterion(y_hat, y.long())  # y must be long type
        # ... rest same
```

---

## **Step 4: Modify Evaluation**

### **Current (Binary):**
```python
def evaluate(model, x_parts, y):
    model.eval()
    with torch.no_grad():
        y_hat = model(x_parts).squeeze()
        y_pred = (y_hat > 0.5).int()  # Binary threshold
    acc = accuracy_score(y.cpu(), y_pred.cpu())
    # ...
```

### **Multi-Class:**
```python
def evaluate(model, x_parts, y):
    model.eval()
    with torch.no_grad():
        y_hat = model(x_parts)  # [batch, num_classes]
        y_pred = torch.argmax(y_hat, dim=1)  # Get class with max probability
    acc = accuracy_score(y.cpu(), y_pred.cpu())
    # Can also use classification_report for per-class metrics
    from sklearn.metrics import classification_report
    print(classification_report(y.cpu(), y_pred.cpu()))
```

---

## **Step 5: Attack-Type-Specific Party Actions**

### **Define Actions Per Attack Type:**
```python
# Attack type to party action mapping
attack_type_actions = {
    0: {  # Benign
        "Party 1": "No action needed",
        "Party 2": "No action needed",
        "Party 3": "No action needed"
    },
    1: {  # DDoS
        "Party 1": "adjust flow timeouts / RAN rate limits",
        "Party 2": "reconfigure load balancer / autoscale VNFs",
        "Party 3": "tighten firewall / ACL on return-path traffic"
    },
    2: {  # PortScan
        "Party 1": "implement port filtering / rate limiting",
        "Party 2": "enable port scan detection / alerting",
        "Party 3": "block suspicious source IPs / firewall rules"
    },
    3: {  # BruteForce
        "Party 1": "implement connection rate limits",
        "Party 2": "enable brute force detection / CAPTCHA",
        "Party 3": "block failed login attempts / IP blacklist"
    },
    4: {  # Bot
        "Party 1": "detect bot traffic patterns",
        "Party 2": "implement bot mitigation / filtering",
        "Party 3": "block bot command & control traffic"
    }
}

# Attack type names
attack_type_names = {
    0: "Benign",
    1: "DDoS",
    2: "PortScan",
    3: "BruteForce",
    4: "Bot"
}
```

---

## **Step 6: Attack-Type-Specific SHAP Analysis**

### **Modify Section 11:**
```python
# Instead of just attack_mask, create masks for each attack type
attack_types = [1, 2, 3, 4]  # Exclude 0 (benign)

# For each attack type
for attack_type in attack_types:
    attack_mask = (y_explain == attack_type)
    attack_name = attack_type_names[attack_type]
    
    mean_phi_attack, mean_pct_attack = compute_subset_shap(attack_mask)
    
    print(f"\n=== SHAP Party Attributions ({attack_name} flows) ===")
    attack_rows = []
    for i, name in enumerate(party_names):
        a_abs = float(np.asarray(mean_phi_attack[i]))
        a_pct = float(np.asarray(mean_pct_attack[i]))
        print(f"{name}: mean |SHAP| = {a_abs:.6f}, mean contribution = {a_pct*100:5.2f}%")
        attack_rows.append({
            "Party": name,
            "Domain": party_domains[i],
            "Feature_Group": party_feature_groups[i],
            "Mean_abs_SHAP": a_abs,
            "Mean_contrib": a_pct,
            "Attack_Type": attack_name,
            "Suggested_Action": attack_type_actions[attack_type][name]
        })
    
    attack_df = pd.DataFrame(attack_rows)
    attack_df.to_csv(f"vfl_shap_{attack_name.lower()}_summary.csv", index=False)
    
    # Find dominant party for this attack type
    top_party_idx = int(np.nanargmax(mean_pct_attack))
    top_party_name = party_names[top_party_idx]
    top_action = attack_type_actions[attack_type][top_party_name]
    
    print(f"Dominant party for {attack_name}: {top_party_name}")
    print(f"Recommended action: {top_action}")
```

---

## **Step 7: Per-Attack-Type Meta-Model**

### **Option 1: One Meta-Model Per Attack Type**
```python
# Create separate meta-models for each attack type
meta_models = {}
for attack_type in attack_types:
    meta_models[attack_type] = PartyMetaModel()
    # Train separately for each attack type
```

### **Option 2: Multi-Class Meta-Model**
```python
class PartyMetaModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)  # 3 inputs, 5 outputs
    
    def forward(self, x_meta):
        return torch.softmax(self.fc(x_meta), dim=1)  # Multi-class output
```

---

## **Step 8: Attack-Type-Specific SHAP**

### **For Each Attack Type:**
```python
# Compute SHAP for each attack type separately
for attack_type in attack_types:
    # Filter samples of this attack type
    attack_type_mask = (y_explain == attack_type)
    X_explain_type = X_explain[attack_type_mask]
    
    # Compute SHAP for this attack type
    shap_values_type = explainer.shap_values(X_explain_type, nsamples=200)
    
    # Analyze per attack type
    # ... same analysis as before but per attack type
```

---

## **Step 9: Update Section 12 (Per-Flow Examples)**

```python
# In per-flow examples, show attack type
for i in range(min(5, n_explain)):
    label_i = int(y_explain[i])
    attack_type_name = attack_type_names.get(label_i, "Unknown")
    
    print(f"\nFlow {i} (label = {label_i}, type = {attack_type_name}):")
    
    # Get actions for this attack type
    if label_i in attack_type_actions:
        actions = attack_type_actions[label_i]
        for party, action in actions.items():
            print(f"  {party} action: {action}")
```

---

## **Step 10: Update Section 13 (Policy Comparison)**

```python
# Compare policies per attack type
for attack_type in attack_types:
    attack_name = attack_type_names[attack_type]
    attack_mask = (y_true == attack_type)
    
    # Policy 1: Global threshold
    blocked_global = (y_scores[:, attack_type] > 0.5)  # Class-specific threshold
    attack_block_rate = blocked_global[attack_mask].mean()
    
    # Policy 2: SHAP-guided (per attack type)
    # ... similar but attack-type specific
    
    print(f"\n{attack_name} Detection:")
    print(f"  Global policy: {attack_block_rate*100:.2f}%")
    print(f"  SHAP policy: {shap_block_rate*100:.2f}%")
```

---

## **Complete Example: Multi-Class Setup**

```python
# 1. Define attack types
attack_type_names = {
    0: "Benign",
    1: "DDoS",
    2: "PortScan",
    3: "BruteForce",
    4: "Bot"
}

num_classes = len(attack_type_names)

# 2. Create multi-class model
model = VFLModel(
    input_dims=[len(party1_features), len(party2_features), len(party3_features)],
    embed_dim=embed_dim,
    num_classes=num_classes  # Add this
)

# 3. Train with CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
# ... training code

# 4. Analyze per attack type
for attack_type in [1, 2, 3, 4]:  # Exclude benign
    attack_mask = (y_explain == attack_type)
    # ... SHAP analysis per attack type
```

---

## **Summary:**

**Key Changes for Multi-Attack Types:**
1. **Labels**: Use `torch.long` instead of `torch.float32`
2. **Model**: Change output to `num_classes` (softmax instead of sigmoid)
3. **Loss**: Use `CrossEntropyLoss` instead of `BCELoss`
4. **Evaluation**: Use `argmax` instead of threshold
5. **SHAP Analysis**: Separate analysis per attack type
6. **Actions**: Define actions per attack type per party
7. **Policies**: Compare policies per attack type

**Benefits:**
- Understand which party detects which attack type
- Attack-type-specific mitigation actions
- Better insights for different threats
- More actionable recommendations
