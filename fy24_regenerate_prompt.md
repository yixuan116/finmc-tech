# FY24 JSON 格式转换提示词

请将以下 FY24 的 JSON 数据转换为标准的 NVDA Firm Fundamentals 格式。

## 当前问题

现有的 FY24 JSON 存在以下问题：
1. **结构不符合标准**：使用了 `{"FY24": {"Q1": {...}}}` 而不是 `{"NVDA_Firm_Fundamentals": [...]}`
2. **字段命名不一致**：使用了 `gross_margin.gaap` 而不是 `gross_margin_gaap`
3. **数值单位需要转换**：
   - revenue 是百万美元（如 7130），需要转换为实际数值（7130000000）
   - gross_margin 是百分比（如 64.6），需要转换为小数（0.646）
4. **缺少关键字段**：
   - 缺少 `period_end`（季度结束日期）
   - 缺少 `report_date`（财报发布日期）
   - 缺少 `operating_expenses_gaap` 和 `operating_expenses_nongaap`（只有 operating_income）
   - 缺少 `narratives`（ai_demand, product_cycle, strategic_themes）
   - 缺少 `source_files`（press_release, ten_q, transcript）

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

## FY24 原始数据

```json
{
    "FY24": {
      "Q1": {
        "quarter": "Q1 2024",
        "period": "FY24 Q1",
        "revenue": {
          "total": 7130,
          "YoY": -13,
          "QoQ": -31
        },
        "segments": {
          "data_center": 4270,
          "gaming": 2224,
          "professional_visualization": 295,
          "automotive": 296,
          "oem_other": 44
        },
        "gross_margin": {
          "gaap": 64.6,
          "non_gaap": 66.8
        },
        "operating_income": {
          "gaap": 2098,
          "non_gaap": 2694
        },
        "net_income": {
          "gaap": 2043,
          "non_gaap": 2680
        },
        "eps": {
          "gaap": 0.82,
          "non_gaap": 1.09
        },
        "cash_flow": {
          "operating_cash_flow": 2670
        },
        "guidance_next_q": {
          "revenue": 11000,
          "gaap_gross_margin": 68.6,
          "non_gaap_gross_margin": 70.0
        }
      },
      "Q2": {
        "quarter": "Q2 2024",
        "period": "FY24 Q2",
        "revenue": {
          "total": 13507,
          "YoY": 101,
          "QoQ": 89
        },
        "segments": {
          "data_center": 10323,
          "gaming": 2426,
          "professional_visualization": 379,
          "automotive": 253,
          "oem_other": 126
        },
        "gross_margin": {
          "gaap": 70.1,
          "non_gaap": 71.2
        },
        "operating_income": {
          "gaap": 6990,
          "non_gaap": 7942
        },
        "net_income": {
          "gaap": 6179,
          "non_gaap": 7422
        },
        "eps": {
          "gaap": 2.48,
          "non_gaap": 2.70
        },
        "cash_flow": {
          "operating_cash_flow": 6946
        },
        "guidance_next_q": {
          "revenue": 16000,
          "gaap_gross_margin": 71.5,
          "non_gaap_gross_margin": 72.5
        }
      },
      "Q3": {
        "quarter": "Q3 2024",
        "period": "FY24 Q3",
        "revenue": {
          "total": 18120,
          "YoY": 206,
          "QoQ": 34
        },
        "segments": {
          "data_center": 14492,
          "gaming": 2811,
          "professional_visualization": 576,
          "automotive": 261,
          "oem_other":  -20
        },
        "gross_margin": {
          "gaap": 74.0,
          "non_gaap": 75.0
        },
        "operating_income": {
          "gaap": 10679,
          "non_gaap": 11614
        },
        "net_income": {
          "gaap": 9293,
          "non_gaap": 10516
        },
        "eps": {
          "gaap": 3.71,
          "non_gaap": 4.02
        },
        "cash_flow": {
          "operating_cash_flow": 13150
        },
        "guidance_next_q": {
          "revenue": 20000,
          "gaap_gross_margin": 74.5,
          "non_gaap_gross_margin": 75.5
        }
      },
      "Q4": {
        "quarter": "Q4 2024",
        "period": "FY24 Q4",
        "revenue": {
          "total": 22100,
          "YoY": 265,
          "QoQ": 22
        },
        "segments": {
          "data_center": 18400,
          "gaming": 2890,
          "professional_visualization": 463,
          "automotive": 281,
          "oem_other": 75
        },
        "gross_margin": {
          "gaap": 76.0,
          "non_gaap": 76.7
        },
        "operating_income": {
          "gaap": 13062,
          "non_gaap": 13862
        },
        "net_income": {
          "gaap": 12285,
          "non_gaap": 12980
        },
        "eps": {
          "gaap": 4.93,
          "non_gaap": 5.16
        },
        "cash_flow": {
          "operating_cash_flow": 13500
        },
        "guidance_next_q": {
          "revenue": null,
          "gaap_gross_margin": null,
          "non_gaap_gross_margin": null
        }
      }
    }
  }
```

## 转换要求

1. **结构转换**：将 `{"FY24": {"Q1": {...}}}` 转换为 `{"NVDA_Firm_Fundamentals": [{...}, {...}, {...}, {...}]}`，包含 Q1-Q4 四个季度

2. **字段映射**：
   - `fiscal_year`: 2024
   - `fiscal_quarter`: "Q1", "Q2", "Q3", "Q4"
   - `period_end`: 需要根据 FY24 的季度结束日期填写（FY24 Q1 结束日期约为 2024-04-28，Q2 约为 2024-07-28，Q3 约为 2024-10-27，Q4 约为 2025-01-26）
   - `report_date`: 需要根据实际财报发布日期填写（通常为季度结束后约 2-3 周）

3. **数值转换**：
   - `revenue`: 从百万美元转换为实际数值（7130 → 7130000000）
   - `gross_margin_gaap`: 从百分比转换为小数（64.6 → 0.646）
   - `gross_margin_nongaap`: 从百分比转换为小数（66.8 → 0.668）
   - `net_income_gaap`: 从百万美元转换为实际数值（2043 → 2043000000）
   - `net_income_nongaap`: 从百万美元转换为实际数值（2680 → 2680000000）
   - `eps_gaap` 和 `eps_nongaap`: 保持不变

4. **计算缺失字段**：
   - `operating_expenses_gaap` = revenue × (1 - gross_margin_gaap) - operating_income_gaap
   - `operating_expenses_nongaap` = revenue × (1 - gross_margin_nongaap) - operating_income_nongaap

5. **Segment 映射**：
   - 将 `professional_visualization` 映射为 `pro_viz`（如果标准格式需要）
   - 将 `automotive` 保持不变
   - 将 `oem_other` 保持不变
   - 注意：FY24 的 segment 结构与 FY26 可能不完全一致，请根据实际情况调整

6. **添加 narratives**：
   - `ai_demand`: 描述 AI 和数据中心需求情况（FY24 是 AI 爆发年，Hopper H100 需求强劲）
   - `product_cycle`: 描述产品周期（Hopper 架构、H100 GPU 等）
   - `strategic_themes`: 列出 3-5 个战略主题

7. **添加 source_files**：
   - `press_release`: "Q1 24 NVIDIAAn.pdf"（根据季度调整）
   - `ten_q`: "Q1 24 10-Q.pdf"（根据季度调整，Q4 用 10-K）
   - `transcript`: 如果有 earnings call transcript，填写文件名，否则为 null

8. **日期参考**（FY24 季度结束和报告日期）：
   - Q1 FY24: period_end 约 2024-04-28, report_date 约 2024-05-22
   - Q2 FY24: period_end 约 2024-07-28, report_date 约 2024-08-23
   - Q3 FY24: period_end 约 2024-10-27, report_date 约 2024-11-21
   - Q4 FY24: period_end 约 2025-01-26, report_date 约 2025-02-21

请按照以上要求，将 FY24 数据转换为标准格式，输出完整的 JSON。

