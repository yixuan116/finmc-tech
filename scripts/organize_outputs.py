#!/usr/bin/env python3
"""
整理 outputs/ 目录的文件结构

这个脚本会：
1. 创建新的目录结构
2. 移动文件到新位置
3. 保留原文件（使用 shutil.move，但可以先做 dry-run）

使用方法:
    python scripts/organize_outputs.py --dry-run  # 只显示会做什么，不实际移动
    python scripts/organize_outputs.py            # 实际执行移动
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# 文件移动映射
FILE_MOVES = {
    # 数据文件 - fundamentals
    'fundamentals': [
        ('outputs/nvda_firm_fundamentals_master.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_firm_fundamentals_master.json.bak', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2009_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2010_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2011_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2012_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2013_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2014_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2015_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2016_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2017_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2018_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2019_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2020_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2021_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2022_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2023_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2024_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2025_q1_q4.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy2026_q1_q3.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy22.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_fy24.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_25 Q123.json', 'outputs/data/fundamentals/'),
        ('outputs/nvda_26 Q123 25 Q4.json', 'outputs/data/fundamentals/'),
        ('outputs/figs/nvda_fy2023.json', 'outputs/data/fundamentals/'),
        ('outputs/figs/nvdafy21.json', 'outputs/data/fundamentals/'),
    ],
    
    # 数据文件 - cash_flow
    'cash_flow': [
        ('outputs/cash_flow_field_scan_results.csv', 'outputs/data/cash_flow/'),
        ('outputs/cash_flow_field_summary.csv', 'outputs/data/cash_flow/'),
        ('outputs/cash_flow_values_extracted.csv', 'outputs/data/cash_flow/'),
        ('outputs/cash_flow_values_extracted_fixed.csv', 'outputs/data/cash_flow/'),
        ('outputs/cash_flow_values_extracted_with_capex.csv', 'outputs/data/cash_flow/'),
    ],
    
    # 数据文件 - training
    'training': [
        ('outputs/phase2_long_cycle/training_data_extended_10y.csv', 'outputs/data/training/'),
        ('outputs/phase2_long_cycle/training_data_with_cash_flow.csv', 'outputs/data/training/'),
        ('outputs/phase2_long_cycle/training_data_with_macro.csv', 'outputs/data/training/'),
        ('outputs/phase2_long_cycle/features_with_cash_flow.csv', 'outputs/data/training/'),
    ],
    
    # 数据文件 - raw
    'raw': [
        ('outputs/NVDA_data_2010_2025.csv', 'outputs/data/raw/'),
        ('outputs/nvda_revenue_features.csv', 'outputs/data/raw/'),
        ('outputs/revenues_nvda_with_prices.csv', 'outputs/data/raw/'),
    ],
    
    # 特征重要性图片 - short_term
    'fi_plots_short': [
        ('outputs/importance_rf_1y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/importance_rf_3y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/importance_rf_7y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/importance_xgb_1y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/importance_xgb_3y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/importance_xgb_7y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/shap_1y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/shap_3y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/shap_7y.png', 'outputs/feature_importance/plots/short_term/'),
        ('outputs/importance_heatmap_all_horizons.png', 'outputs/feature_importance/plots/short_term/'),
    ],
    
    # 特征重要性图片 - mid_term
    'fi_plots_mid': [
        ('outputs/phase2_long_cycle/feature_importance/feature_importance_by_horizon_heatmap.png', 
         'outputs/feature_importance/plots/mid_term/'),
    ],
    
    # 特征重要性图片 - long_term
    'fi_plots_long': [
        ('outputs/phase2_long_cycle/long_term_feature_importance/feature_importance_long_term_heatmap.png',
         'outputs/feature_importance/plots/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/firm_vs_macro_importance_comparison.png',
         'outputs/feature_importance/plots/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/fcf_importance_long_term_analysis.png',
         'outputs/feature_importance/plots/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/fcf_comprehensive_analysis.png',
         'outputs/feature_importance/plots/long_term/'),
    ],
    
    # 特征重要性数据 - mid_term
    'fi_data_mid': [
        ('outputs/phase2_long_cycle/feature_importance/feature_importance_12q_detailed.csv',
         'outputs/feature_importance/data/mid_term/'),
    ],
    
    # 特征重要性数据 - long_term
    'fi_data_long': [
        ('outputs/phase2_long_cycle/long_term_feature_importance/feature_importance_long_term_all_features.csv',
         'outputs/feature_importance/data/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/feature_importance_y_log_20q_all_features.csv',
         'outputs/feature_importance/data/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/feature_importance_y_log_28q_all_features.csv',
         'outputs/feature_importance/data/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/feature_importance_y_log_40q_all_features.csv',
         'outputs/feature_importance/data/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/firm_vs_macro_importance_summary.csv',
         'outputs/feature_importance/data/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/fcf_importance_by_horizon.csv',
         'outputs/feature_importance/data/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/fcf_importance_complete_comparison.csv',
         'outputs/feature_importance/data/long_term/'),
        ('outputs/phase2_long_cycle/long_term_feature_importance/README.md',
         'outputs/feature_importance/data/long_term/'),
    ],
    
    # 通用可视化
    'figs_general': [
        ('outputs/figs/pred_vs_actual.png', 'outputs/figs/general/'),
        ('outputs/figs/pred_vs_actual_price_direct.png', 'outputs/figs/general/'),
        ('outputs/figs/pred_vs_actual_price_indirect.png', 'outputs/figs/general/'),
        ('outputs/figs/pred_vs_actual_return_rf.png', 'outputs/figs/general/'),
        ('outputs/figs/residuals_return.png', 'outputs/figs/general/'),
        ('outputs/figs/calibration_return.png', 'outputs/figs/general/'),
        ('outputs/figs/accel_vs_return.png', 'outputs/figs/general/'),
        ('outputs/figs/yoy_vs_return.png', 'outputs/figs/general/'),
        ('outputs/figs/rolling_corr.png', 'outputs/figs/general/'),
        ('outputs/figs/rf_feature_importance.png', 'outputs/figs/general/'),
        ('outputs/comparison_ml_vs_mc.png', 'outputs/figs/general/'),
        ('outputs/comparison_yearly.png', 'outputs/figs/general/'),
        ('outputs/ml_baseline.png', 'outputs/figs/general/'),
    ],
    
    # 结果文件
    'results': [
        ('outputs/evaluation_table.csv', 'outputs/results/'),
        ('outputs/results_all.csv', 'outputs/results/'),
        ('outputs/results_forecast.csv', 'outputs/results/'),
        ('outputs/results_mc.csv', 'outputs/results/'),
        ('outputs/nvda_ml_pred.csv', 'outputs/results/'),
        ('outputs/nvda_mc_terminals.csv', 'outputs/results/'),
        ('outputs/nvda_mc_meta.json', 'outputs/results/'),
    ],
}


def organize_files(dry_run: bool = True):
    """整理文件"""
    base_path = Path('.')
    
    total_files = 0
    moved_files = 0
    skipped_files = 0
    errors = []
    
    print("=" * 80)
    print("文件整理" + (" (DRY RUN - 只显示，不实际移动)" if dry_run else " (实际执行)"))
    print("=" * 80)
    
    for category, moves in FILE_MOVES.items():
        print(f"\n📁 {category}:")
        
        for src, dst_dir in moves:
            src_path = base_path / src
            dst_path = base_path / dst_dir / src_path.name
            
            total_files += 1
            
            if not src_path.exists():
                print(f"  ⚠️  跳过 (不存在): {src}")
                skipped_files += 1
                continue
            
            if dst_path.exists():
                print(f"  ⚠️  跳过 (目标已存在): {src} → {dst_path}")
                skipped_files += 1
                continue
            
            if dry_run:
                print(f"  📋 将移动: {src} → {dst_path}")
            else:
                try:
                    # 创建目标目录
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    # 移动文件
                    shutil.move(str(src_path), str(dst_path))
                    print(f"  ✓ 已移动: {src} → {dst_path}")
                    moved_files += 1
                except Exception as e:
                    print(f"  ✗ 错误: {src} → {e}")
                    errors.append((src, str(e)))
                    skipped_files += 1
    
    print("\n" + "=" * 80)
    print("总结:")
    print(f"  总文件数: {total_files}")
    print(f"  已处理: {moved_files}")
    print(f"  跳过: {skipped_files}")
    if errors:
        print(f"  错误: {len(errors)}")
        for src, err in errors:
            print(f"    - {src}: {err}")
    print("=" * 80)
    
    if dry_run:
        print("\n💡 这是 DRY RUN，没有实际移动文件")
        print("   运行 python scripts/organize_outputs.py 来实际执行")


def main():
    parser = argparse.ArgumentParser(description="整理 outputs/ 目录的文件结构")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='只显示会做什么，不实际移动文件（默认）'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='实际执行文件移动（需要明确指定）'
    )
    
    args = parser.parse_args()
    
    # 如果指定了 --execute，则实际执行
    dry_run = not args.execute
    
    organize_files(dry_run=dry_run)


if __name__ == '__main__':
    main()

