#!/usr/bin/env python3
"""
自动更新代码中的路径引用（整理文件后使用）

这个脚本会自动更新所有 Python 文件中的路径引用，匹配新的目录结构。
"""

import re
from pathlib import Path
from typing import List, Tuple

# 路径映射规则（旧路径 -> 新路径）
PATH_MAPPINGS = [
    # 数据文件
    (r'outputs/nvda_firm_fundamentals_master\.json', 'outputs/data/fundamentals/nvda_firm_fundamentals_master.json'),
    (r'outputs/cash_flow_([^/"]+)\.csv', r'outputs/data/cash_flow/cash_flow_\1.csv'),
    (r'outputs/phase2_long_cycle/training_data_([^/"]+)\.csv', r'outputs/data/training/training_data_\1.csv'),
    (r'outputs/phase2_long_cycle/features_with_cash_flow\.csv', 'outputs/data/training/features_with_cash_flow.csv'),
    (r'outputs/NVDA_data_2010_2025\.csv', 'outputs/data/raw/NVDA_data_2010_2025.csv'),
    (r'outputs/nvda_revenue_features\.csv', 'outputs/data/raw/nvda_revenue_features.csv'),
    (r'outputs/revenues_nvda_with_prices\.csv', 'outputs/data/raw/revenues_nvda_with_prices.csv'),
    
    # 特征重要性图片 - short_term
    (r'outputs/importance_rf_([137]y)\.png', r'outputs/feature_importance/plots/short_term/importance_rf_\1.png'),
    (r'outputs/importance_xgb_([137]y)\.png', r'outputs/feature_importance/plots/short_term/importance_xgb_\1.png'),
    (r'outputs/shap_([137]y)\.png', r'outputs/feature_importance/plots/short_term/shap_\1.png'),
    (r'outputs/importance_heatmap_all_horizons\.png', 'outputs/feature_importance/plots/short_term/importance_heatmap_all_horizons.png'),
    
    # 特征重要性图片 - mid_term
    (r'outputs/feature_importance/data/mid_term/feature_importance_by_horizon_heatmap\.png',
     'outputs/feature_importance/plots/mid_term/feature_importance_by_horizon_heatmap.png'),
    
    # 特征重要性图片 - long_term
    (r'outputs/feature_importance/data/long_term/feature_importance_long_term_heatmap\.png',
     'outputs/feature_importance/plots/long_term/feature_importance_long_term_heatmap.png'),
    (r'outputs/feature_importance/data/long_term/firm_vs_macro_importance_comparison\.png',
     'outputs/feature_importance/plots/long_term/firm_vs_macro_importance_comparison.png'),
    (r'outputs/feature_importance/data/long_term/fcf_importance_long_term_analysis\.png',
     'outputs/feature_importance/plots/long_term/fcf_importance_long_term_analysis.png'),
    (r'outputs/feature_importance/data/long_term/fcf_comprehensive_analysis\.png',
     'outputs/feature_importance/plots/long_term/fcf_comprehensive_analysis.png'),
    
    # 特征重要性数据 - mid_term
    (r'outputs/feature_importance/data/mid_term/feature_importance_12q_detailed\.csv',
     'outputs/feature_importance/data/mid_term/feature_importance_12q_detailed.csv'),
    
    # 特征重要性数据 - long_term
    (r'outputs/feature_importance/data/long_term/feature_importance_long_term_all_features\.csv',
     'outputs/feature_importance/data/long_term/feature_importance_long_term_all_features.csv'),
    (r'outputs/feature_importance/data/long_term/feature_importance_y_log_(\d+q)_all_features\.csv',
     r'outputs/feature_importance/data/long_term/feature_importance_y_log_\1_all_features.csv'),
    (r'outputs/feature_importance/data/long_term/firm_vs_macro_importance_summary\.csv',
     'outputs/feature_importance/data/long_term/firm_vs_macro_importance_summary.csv'),
    (r'outputs/feature_importance/data/long_term/fcf_importance_by_horizon\.csv',
     'outputs/feature_importance/data/long_term/fcf_importance_by_horizon.csv'),
    (r'outputs/feature_importance/data/long_term/fcf_importance_complete_comparison\.csv',
     'outputs/feature_importance/data/long_term/fcf_importance_complete_comparison.csv'),
    (r'outputs/feature_importance/data/long_term/README\.md',
     'outputs/feature_importance/data/long_term/README.md'),
    
    # 特征重要性目录引用
    (r'outputs/feature_importance/data/long_term', 'outputs/feature_importance/data/long_term'),
    (r'outputs/feature_importance/data/mid_term', 'outputs/feature_importance/data/mid_term'),
    
    # 通用可视化（保持 figs/ 但整理到 general/）
    (r'outputs/comparison_ml_vs_mc\.png', 'outputs/figs/general/comparison_ml_vs_mc.png'),
    (r'outputs/comparison_yearly\.png', 'outputs/figs/general/comparison_yearly.png'),
    (r'outputs/ml_baseline\.png', 'outputs/figs/general/ml_baseline.png'),
]


def update_file_paths(file_path: Path, dry_run: bool = True) -> List[Tuple[str, str]]:
    """更新单个文件中的路径引用"""
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        changes = []
        
        for pattern, replacement in PATH_MAPPINGS:
            # 使用正则表达式替换
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                # 找到所有匹配
                matches = re.finditer(pattern, content)
                for match in matches:
                    old_path = match.group(0)
                    new_path = re.sub(pattern, replacement, old_path)
                    changes.append((old_path, new_path))
                content = new_content
        
        if not dry_run and content != original_content:
            file_path.write_text(content, encoding='utf-8')
        
        return changes
    except Exception as e:
        print(f"  ✗ 错误处理 {file_path}: {e}")
        return []


def find_python_files() -> List[Path]:
    """查找所有需要检查的 Python 文件"""
    files = []
    for pattern in ['*.py', '*.ipynb']:
        for file_path in Path('.').rglob(pattern):
            # 跳过某些目录
            if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.ipynb_checkpoints', 'outputs']):
                continue
            files.append(file_path)
    return files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="更新代码中的路径引用")
    parser.add_argument('--dry-run', action='store_true', default=True, help='只显示会做什么（默认）')
    parser.add_argument('--execute', action='store_true', help='实际执行更新')
    
    args = parser.parse_args()
    dry_run = not args.execute
    
    print("=" * 80)
    print("路径引用更新" + (" (DRY RUN)" if dry_run else " (实际执行)"))
    print("=" * 80)
    
    files = find_python_files()
    total_changes = 0
    files_changed = 0
    
    for file_path in files:
        changes = update_file_paths(file_path, dry_run=dry_run)
        if changes:
            files_changed += 1
            total_changes += len(changes)
            print(f"\n📄 {file_path}")
            for old, new in changes[:5]:  # 只显示前5个
                print(f"  {old}")
                print(f"  → {new}")
            if len(changes) > 5:
                print(f"  ... 还有 {len(changes) - 5} 处更改")
    
    print("\n" + "=" * 80)
    print(f"总结: {files_changed} 个文件，{total_changes} 处路径更新")
    if dry_run:
        print("\n💡 这是 DRY RUN，没有实际修改文件")
        print("   运行 python scripts/update_paths_after_organize.py --execute 来实际执行")


if __name__ == '__main__':
    main()

