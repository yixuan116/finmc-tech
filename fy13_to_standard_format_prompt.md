# FY13 转换为标准格式提示词

请将以下 FY13 的 JSON 数据转换为标准的 NVDA Firm Fundamentals 格式，使其与 FY26 的格式完全一致。

## 当前问题

FY13 的 JSON 存在以下结构不一致问题：

1. **Q1-Q3 与 Q4 结构不同**：
   - Q1-Q3 使用 `gaap`/`non_gaap` 作为顶层字段
   - Q4 使用 `financials.gaap`/`financials.non_gaap`
   - 需要统一为标准的 `financials` 结构

2. **字段命名不一致**：
   - Q1-Q3: `gross_margin_percent`, `revenue_musd`, `eps_diluted_usd`
   - Q4: `gross_margin_pct`, `revenue_musd`, `eps_diluted_usd`
   - 需要统一为标准格式的命名

3. **数值单位需要转换**：
   - `revenue_musd`（百万美元）→ `revenue`（实际数值，乘以 1,000,000）
   - `gross_margin_percent`（百分比）→ `gross_margin_gaap`（小数，除以 100）

4. **缺少关键字段**：
   - Q1-Q3 缺少 `report_date`
   - Q4 缺少完整的顶层字段（`fiscal_year`, `fiscal_quarter`, `period_end`, `report_date`）
   - Q4 缺少 `narratives` 和 `source_files`

5. **narratives 结构不一致**：
   - Q1-Q3 使用 `ceo_commentary`, `strategy_highlights`, `products_and_platforms`
   - 需要转换为标准的 `ai_demand`, `product_cycle`, `strategic_themes`

## 标准格式参考（FY26）

```json
{
  "NVDA_Firm_Fundamentals": [
    {
      "fiscal_year": 2026,
      "fiscal_quarter": "Q1",
      "period_end": "2025-04-27",
      "report_date": "2025-05-28",
      "financials": {
        "revenue": 49280000000,
        "gross_margin_gaap": 0.765,
        "gross_margin_nongaap": 0.772,
        "operating_expenses_gaap": 4509000000,
        "operating_expenses_nongaap": 3331000000,
        "net_income_gaap": 26410000000,
        "net_income_nongaap": 27190000000,
        "eps_gaap": 1.05,
        "eps_nongaap": 1.08,
        "segment": {
          "data_center": 42800000000,
          "client": 2200000000,
          "gaming": 3000000000
        }
      },
      "narratives": {
        "ai_demand": "Hopper demand remains strong; early Blackwell revenue expected next quarter.",
        "product_cycle": "Blackwell adoption in early phases.",
        "strategic_themes": [
          "AI supercomputing",
          "Enterprise AI",
          "AI infra capex growth"
        ]
      },
      "source_files": {
        "press_release": "Q1 26 NVIDIAAn.pdf",
        "ten_q": "Q1 26 10-Q.pdf",
        "transcript": "NVDA-Q1-2026-Earnings-Call-28-May-2025-5_00-PM-ET.pdf"
      }
    }
  ]
}
```

## FY13 原始数据

```json
{
    "fiscal_year": 2013,
    "fiscal_year_end": "2013-01-27",
    "currency": "USD",
    "press_release_date": "2013-02-13",
    "title": "NVIDIA Reports Financial Results for Annual and Fourth Quarter Fiscal 2013",
    "summary_bullets": [
      "Full-year revenue increased 7.1 percent to a record $4.28 billion.",
      "Quarterly revenue decreased 8.1 percent sequentially to $1.11 billion; year on year, revenue was up 16.1 percent.",
      "Quarterly GAAP diluted EPS was $0.28 vs. $0.33 in Q3; non-GAAP diluted EPS was $0.35 vs. $0.39 in Q3.",
      "Quarterly GAAP gross margin was 52.9%; non-GAAP gross margin was 53.2%."
    ],
  
    "annual": {
      "financials": {
        "gaap": {
          "revenue_musd": 4280.2,
          "revenue_yoy_pct": 7.1,
          "gross_margin_pct": 52.0,
          "operating_expenses_musd": 1578.1,
          "operating_expenses_yoy_pct": 12.1,
          "operating_income_musd": 648.2,
          "net_income_musd": 562.5,
          "net_income_yoy_pct": -3.2,
          "eps_diluted_usd": 0.90,
          "eps_diluted_yoy_pct": -4.3,
          "effective_tax_rate_pct": 15.0
        },
        "non_gaap": {
          "revenue_musd": 4280.2,
          "revenue_yoy_pct": 7.1,
          "gross_margin_pct": 52.3,
          "operating_expenses_musd": 1395.7,
          "operating_expenses_yoy_pct": 12.0,
          "net_income_musd": 728.4,
          "net_income_yoy_pct": -0.8,
          "eps_diluted_usd": 1.17,
          "eps_diluted_yoy_pct": -1.7,
          "effective_tax_rate_pct": 14.8
        }
      },
      "segments": {
        "annual_revenue_musd": {
          "gpu": {
            "revenue_musd": 3251.7,
            "revenue_yoy_pct": 2.0,
            "comment": "GPU business grew despite chipset wind-down; underlying GPU ex-chipset grew ~8%."
          },
          "tegra_processor": {
            "revenue_musd": 764.5,
            "revenue_yoy_pct": 29.3,
            "smartphone_tablet_revenue_musd": 540.5,
            "smartphone_tablet_yoy_pct": 50.3,
            "comment": "Growth driven by Tegra 3 smartphones/tablets and automotive."
          },
          "all_other": {
            "revenue_musd": 264.0,
            "revenue_yoy_pct": 20.0,
            "comment": "Primarily Intel patent cross-license royalty revenue (full-year vs partial in FY12)."
          },
          "total_revenue_musd": 4280.2
        },
        "segment_structure_change": {
          "effective_quarter": "Q4 FY2013",
          "old_segments": [
            "GPU",
            "Professional Solutions Business",
            "Consumer Products Business"
          ],
          "new_segments": [
            "GPU",
            "Tegra Processor"
          ],
          "reason": "Align external reporting with internal management of businesses."
        }
      },
      "capital_return": {
        "share_repurchase_q4_musd": 100.0,
        "dividend_per_share_usd": 0.075,
        "dividend_cash_paid_q4_musd": 46.9,
        "comment": "Company continued stock repurchases and quarterly cash dividend program, primarily executed in Q4."
      },
      "tax_and_margins": {
        "annual_gross_margin_gaap_pct": 52.0,
        "annual_gross_margin_non_gaap_pct": 52.3,
        "annual_effective_tax_rate_gaap_pct": 15.0,
        "annual_effective_tax_rate_non_gaap_pct": 14.8
      }
    },
  
    "quarters": {
      "Q1": {
        "fiscal_year": 2013,
        "fiscal_quarter": "Q1",
        "period_end": "2012-04-29",
        "filing_type": ["10-Q", "press_release"],
        "gaap": {
          "revenue_musd": 924.9,
          "gross_profit_musd": 463.4,
          "gross_margin_percent": 50.1,
          "operating_expenses_musd": 390.5,
          "operating_income_musd": 72.8,
          "net_income_musd": 60.4,
          "eps_diluted_usd": 0.10
        },
        "non_gaap": {
          "revenue_musd": 924.9,
          "gross_profit_musd": 465.9,
          "gross_margin_percent": 50.4,
          "operating_expenses_musd": 348.0,
          "operating_income_musd": null,
          "net_income_musd": 97.5,
          "eps_diluted_usd": 0.16
        },
        "guidance": null,
        "narratives": {
          "ceo_commentary": null,
          "strategy_highlights": [],
          "products_and_platforms": [],
          "capital_return": []
        },
        "segment": null,
        "source_files": null
      },
  
      "Q2": {
        "fiscal_year": 2013,
        "fiscal_quarter": "Q2",
        "period_end": "2012-07-29",
        "filing_type": ["10-Q", "press_release"],
        "gaap": {
          "revenue_musd": 1044.3,
          "gross_profit_musd": 540.7,
          "gross_margin_percent": 51.8,
          "operating_expenses_musd": 401.1,
          "operating_income_musd": 139.6,
          "net_income_musd": 119.0,
          "eps_diluted_usd": 0.19
        },
        "non_gaap": {
          "revenue_musd": 1044.3,
          "gross_profit_musd": 543.4,
          "gross_margin_percent": 52.0,
          "operating_expenses_musd": 342.5,
          "operating_income_musd": null,
          "net_income_musd": 170.4,
          "eps_diluted_usd": 0.27
        },
        "guidance": {
          "next_quarter_revenue_range_musd": [1150.0, 1250.0],
          "next_quarter_gross_margin_gaap_percent": 51.8,
          "next_quarter_gross_margin_non_gaap_percent": 52.0,
          "next_quarter_opex_gaap_musd": 390.0,
          "next_quarter_opex_non_gaap_musd": 350.0
        },
        "narratives": {
          "ceo_commentary": "Investments in mobile computing and visual computing are paying off; Tegra reached record sales as tablets scaled, and GPUs made strong gains in a weak market driven by the Kepler architecture.",
          "strategy_highlights": [
            "Positioning at the center of fast-growing segments of mobile and visual computing."
          ],
          "products_and_platforms": [
            "Tegra 3 adoption in tablets including Nexus 7.",
            "Kepler-based GPUs ramping in notebooks and desktops."
          ],
          "capital_return": []
        },
        "segment": null,
        "source_files": null
      },
  
      "Q3": {
        "fiscal_year": 2013,
        "fiscal_quarter": "Q3",
        "period_end": "2012-10-28",
        "filing_type": ["10-Q", "press_release"],
        "gaap": {
          "revenue_musd": 1204.1,
          "gross_profit_musd": 636.7,
          "gross_margin_percent": 52.9,
          "operating_expenses_musd": 384.4,
          "operating_income_musd": 252.2,
          "net_income_musd": 209.1,
          "eps_diluted_usd": 0.33
        },
        "non_gaap": {
          "revenue_musd": 1204.1,
          "gross_profit_musd": 639.1,
          "gross_margin_percent": 53.1,
          "operating_expenses_musd": 344.8,
          "operating_income_musd": null,
          "net_income_musd": 245.5,
          "eps_diluted_usd": 0.39
        },
        "guidance": {
          "next_quarter_revenue_range_musd": [1025.0, 1175.0],
          "next_quarter_gross_margin_gaap_percent": 52.9,
          "next_quarter_gross_margin_non_gaap_percent": 53.1,
          "next_quarter_opex_gaap_musd": 400.0,
          "next_quarter_opex_non_gaap_musd": 359.0
        },
        "narratives": {
          "ceo_commentary": "Investments in new growth strategies delivered record revenue and margins; Kepler GPUs won across gaming, design and supercomputing, while Tegra powered innovative tablets, phones and cars.",
          "strategy_highlights": [
            "Launched and scaled Kepler across gaming, workstation and supercomputing.",
            "Expanded mobile footprint with Tegra in flagship tablets and automotive."
          ],
          "products_and_platforms": [
            "Kepler-based GeForce 600 series GPUs.",
            "Kepler-based Tesla GPUs powering Titan, the world's fastest open-science supercomputer.",
            "VGX K2 GPU for cloud-based workstation graphics."
          ],
          "capital_return": [
            "Initiated a quarterly cash dividend of $0.075 per share.",
            "Extended the $2.7B share repurchase program through December 2014."
          ]
        },
        "segment": null,
        "source_files": null
      },
  
      "Q4": {
        "financials": {
          "gaap": {
            "quarter_label": "Q4 FY2013",
            "revenue_musd": 1106.9,
            "revenue_qoq_pct": -8.1,
            "revenue_yoy_pct": 16.1,
            "gross_margin_pct": 52.9,
            "operating_expenses_musd": 402.0,
            "operating_expenses_qoq_pct": 4.6,
            "operating_expenses_yoy_pct": 9.3,
            "net_income_musd": 174.0,
            "net_income_qoq_pct": -16.8,
            "net_income_yoy_pct": 50.0,
            "eps_diluted_usd": 0.28,
            "eps_diluted_qoq_pct": -15.2,
            "eps_diluted_yoy_pct": 47.4,
            "effective_tax_rate_pct": 6.5
          },
          "non_gaap": {
            "revenue_musd": 1106.9,
            "revenue_qoq_pct": -8.1,
            "revenue_yoy_pct": 16.1,
            "gross_margin_pct": 53.2,
            "operating_expenses_musd": 360.4,
            "operating_expenses_qoq_pct": 4.5,
            "operating_expenses_yoy_pct": 10.8,
            "net_income_musd": 214.9,
            "net_income_qoq_pct": -12.5,
            "net_income_yoy_pct": 35.9,
            "eps_diluted_usd": 0.35,
            "eps_diluted_qoq_pct": -10.3,
            "eps_diluted_yoy_pct": 34.6,
            "effective_tax_rate_estimate_pct": 6.8
          }
        }
      }
    },
  
    "narratives": {
      "ceo_commentary": "We achieved record revenues, margins and cash despite significant market headwinds. We grew our GPU and Tegra Processor businesses and created new pillars for long term growth with Project SHIELD and NVIDIA GRID.",
      "growth_pillars": [
        "Kepler-generation GeForce, Quadro, and Tesla GPUs.",
        "Tegra mobile/automotive platforms including Tegra 3 and Tegra 4.",
        "Project SHIELD and NVIDIA GRID as new product categories in mobile and cloud gaming."
      ]
    },
  
    "source_urls": [
      "https://nvidianews.nvidia.com/news/nvidia-reports-financial-results-for-annual-and-fourth-quarter-fiscal-2013",
      "https://www.sec.gov/Archives/edgar/data/1045810/000104581013000003/q413cfocommentary.htm"
    ]
  }
```

## 转换规则

### 1. 结构转换
- 将 `quarters.Q1`, `quarters.Q2`, `quarters.Q3`, `quarters.Q4` 转换为 `NVDA_Firm_Fundamentals` 数组
- 每个季度作为一个独立对象，包含完整的字段

### 2. 字段映射（Q1-Q3）

**顶层字段**：
- `fiscal_year`: 从 `quarters.Q1.fiscal_year` 获取（2013）
- `fiscal_quarter`: 从 `quarters.Q1.fiscal_quarter` 获取（"Q1", "Q2", "Q3"）
- `period_end`: 从 `quarters.Q1.period_end` 获取（已有）
- `report_date`: 需要根据 `period_end` 推算（通常为季度结束后 2-3 周）

**financials 字段**：
- `revenue`: `gaap.revenue_musd × 1,000,000`（转换为实际数值）
- `gross_margin_gaap`: `gaap.gross_margin_percent / 100`（转换为小数）
- `gross_margin_nongaap`: `non_gaap.gross_margin_percent / 100`（转换为小数）
- `operating_expenses_gaap`: `gaap.operating_expenses_musd × 1,000,000`（转换为实际数值）
- `operating_expenses_nongaap`: `non_gaap.operating_expenses_musd × 1,000,000`（转换为实际数值）
- `net_income_gaap`: `gaap.net_income_musd × 1,000,000`（转换为实际数值）
- `net_income_nongaap`: `non_gaap.net_income_musd × 1,000,000`（转换为实际数值）
- `eps_gaap`: `gaap.eps_diluted_usd`（保持不变）
- `eps_nongaap`: `non_gaap.eps_diluted_usd`（保持不变）
- `segment`: 从 `quarters.Q1.segment` 获取，如果为 `null` 则设为 `null`

### 3. 字段映射（Q4）

**顶层字段**：
- `fiscal_year`: 2013
- `fiscal_quarter`: "Q4"
- `period_end`: "2013-01-27"（从顶层 `fiscal_year_end` 获取）
- `report_date`: "2013-02-13"（从顶层 `press_release_date` 获取）

**financials 字段**：
- `revenue`: `financials.gaap.revenue_musd × 1,000,000`
- `gross_margin_gaap`: `financials.gaap.gross_margin_pct / 100`
- `gross_margin_nongaap`: `financials.non_gaap.gross_margin_pct / 100`
- `operating_expenses_gaap`: `financials.gaap.operating_expenses_musd × 1,000,000`
- `operating_expenses_nongaap`: `financials.non_gaap.operating_expenses_musd × 1,000,000`
- `net_income_gaap`: `financials.gaap.net_income_musd × 1,000,000`
- `net_income_nongaap`: `financials.non_gaap.net_income_musd × 1,000,000`
- `eps_gaap`: `financials.gaap.eps_diluted_usd`
- `eps_nongaap`: `financials.non_gaap.eps_diluted_usd`
- `segment`: 设为 `null`（Q4 没有细分数据）

### 4. narratives 转换

**Q1-Q3**：
- 从 `quarters.Q1.narratives` 提取：
  - `ai_demand`: 从 `ceo_commentary` 或 `products_and_platforms` 中提取与 AI/HPC/Tesla 相关的内容，如果没有则从顶层 `narratives.ceo_commentary` 或 `annual.disaggregation_notes` 提取
  - `product_cycle`: 从 `ceo_commentary` 或 `products_and_platforms` 提取产品周期信息
  - `strategic_themes`: 从 `strategy_highlights` 转换，如果没有则从顶层 `narratives.growth_pillars` 提取

**Q4**：
- 从顶层 `narratives` 提取：
  - `ai_demand`: 从 `ceo_commentary` 或 `annual.disaggregation_notes.gpu_drivers` 中提取与 Tesla/HPC 相关的内容
  - `product_cycle`: 从 `ceo_commentary` 提取
  - `strategic_themes`: 从 `growth_pillars` 提取

### 5. source_files 转换

**Q1-Q3**：
- 从 `quarters.Q1.filing_type` 推断：
  - `press_release`: "Q1 13 NVIDIAAn.pdf"（根据季度调整）
  - `ten_q`: "Q1 13 10-Q.pdf"（根据季度调整）
  - `transcript`: `null`（如果没有 transcript 文件）

**Q4**：
- 从顶层 `source_urls` 推断：
  - `press_release`: "Q4 13 NVIDIAAn.pdf"
  - `ten_q`: `null`（Q4 通常用 10-K）
  - `ten_k`: "Q4 13 10-K.pdf"（如果有）
  - `transcript`: `null`

### 6. report_date 推算（Q1-Q3）

根据 `period_end` 推算：
- Q1 FY13: `period_end` = "2012-04-29"，`report_date` 约 "2012-05-10"（约 11 天后）
- Q2 FY13: `period_end` = "2012-07-29"，`report_date` 约 "2012-08-09"（约 11 天后）
- Q3 FY13: `period_end` = "2012-10-28"，`report_date` 约 "2012-11-08"（约 11 天后）

### 7. 日期格式
- 所有日期必须使用 `YYYY-MM-DD` 格式
- `period_end` 和 `report_date` 必须为有效日期字符串

## 输出要求

1. 输出完整的 JSON，包含 Q1-Q4 四个季度
2. 严格按照标准格式（FY26）的结构
3. 所有数值必须正确转换（百万美元 → 实际数值，百分比 → 小数）
4. 所有字段必须存在，缺失的字段设为 `null`
5. 输出有效的 JSON，可以直接使用 `json.loads()` 解析

## 注意事项

- 保持数据准确性，不要修改原始数值
- 如果某个字段在原始数据中不存在，设为 `null`
- `segment` 字段：如果原始数据中为 `null`，保持为 `null`
- `narratives` 字段：尽量从原始数据中提取，如果无法提取则使用合理的描述
- `source_files` 字段：根据 `filing_type` 或 `source_urls` 推断文件名格式

请按照以上规则，将 FY13 数据转换为标准格式，输出完整的 JSON。

