# Interaction Features Generation Diagram

## High-Level Structure: Macro × Firm = Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                    MACRO FEATURES (4)                            │
├─────────────────────────────────────────────────────────────────┤
│  vix_level    tnx_yield    vix_change_3m    tnx_change_3m      │
└─────────────────────────────────────────────────────────────────┘
                              ×
┌─────────────────────────────────────────────────────────────────┐
│                    FIRM FEATURES (10)                            │
├─────────────────────────────────────────────────────────────────┤
│  rev_yoy      rev_qoq      rev_accel      revenue               │
│  price_returns_1m    price_returns_3m    price_returns_6m       │
│  price_returns_12m   price_momentum      price_volatility        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              INTERACTION FEATURES (4 × 10 = 40)                  │
├─────────────────────────────────────────────────────────────────┤
│  ix_vix_level__rev_yoy                                           │
│  ix_vix_level__rev_qoq                                           │
│  ix_vix_level__rev_accel                                         │
│  ...                                                              │
│  ix_tnx_yield__price_returns_12m                                 │
│  ix_vix_change_3m__price_volatility                              │
│  ...                                                              │
│  (All 40 combinations)                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Matrix Representation

```
        Macro (4 features)
        ┌─────┬─────┬─────┬─────┐
        │ vix │ tnx │ vix │ tnx │
        │_lev │_yld │_chg │_chg │
        └─────┴─────┴─────┴─────┘
Firm    │     │     │     │     │
(10)    │  ×  │  ×  │  ×  │  ×  │
        │     │     │     │     │
        └─────┴─────┴─────┴─────┘
           ↓     ↓     ↓     ↓
        ┌─────┬─────┬─────┬─────┐
        │ ix_ │ ix_ │ ix_ │ ix_ │
        │ ... │ ... │ ... │ ... │
        │ (10)│ (10)│ (10)│ (10)│
        └─────┴─────┴─────┴─────┘
         Total: 4 × 10 = 40 Interaction Features
```

## Example: How One Interaction is Created

```
vix_level (Macro)  ×  rev_yoy (Firm)  =  ix_vix_level__rev_yoy (Interaction)
     │                    │                         │
     │                    │                         │
     │                    │                         │
  [0.25]              [0.15]                   [0.0375]
  (VIX=25)         (Revenue +15%)          (State-dependent effect)
```

## Final Feature Space Structure

```
┌─────────────────────────────────────────────────────────────┐
│                    TOTAL FEATURE SPACE                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Firm Features:        12 features                           │
│  Macro Features:       4 features                            │
│  Interaction Features: 40 features (4 × 10)                  │
│  Time Features:        4 features                             │
│  Metadata:            3 features                              │
│                                                               │
│  Total:               63 features                             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

