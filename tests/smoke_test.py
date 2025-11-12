"""
Smoke test for the complete pipeline.

Runs a truncated version of the pipeline to verify all components work.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from finmc_tech.config import Settings
from finmc_tech.sim.run_simulation import pipeline


def test_pipeline():
    """Run smoke test with truncated date window."""
    print("=" * 70)
    print("Running Smoke Test")
    print("=" * 70)
    
    # Override config with truncated dates for faster testing
    test_config = Settings(
        TICKER="NVDA",
        START_DATE="2015-01-01",
        END_DATE="2022-12-31",
        TRAIN_END="2021-12-31",
        TEST_START="2022-01-01",
        CACHE_DIR="data_cache",
        RESULTS_DIR="results_test",
        RANDOM_STATE=42,
    )
    
    # Run pipeline with small parameters for speed
    print("\nRunning pipeline with test configuration...")
    results = pipeline(
        config=test_config,
        H=12,  # 12 months horizon
        n_paths=50,  # Fewer paths for speed
        shock="base",
    )
    
    # Assertions
    print("\n" + "=" * 70)
    print("Running Assertions")
    print("=" * 70)
    
    # 1. Check model file exists
    model_path = results["model_path"]
    assert model_path.exists(), f"Model file not found: {model_path}"
    assert model_path.stat().st_size > 0, f"Model file is empty: {model_path}"
    print(f"✓ Model file exists: {model_path}")
    
    # 2. Check metrics contain required keys
    metrics = results["metrics"]
    required_keys = ["R2", "MAE", "RMSE"]
    for key in required_keys:
        assert key in metrics, f"Metrics missing key: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} is not numeric"
    print(f"✓ Metrics contain required keys: {required_keys}")
    print(f"  R²: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}")
    
    # 3. Check simulation CSV files exist and are non-empty
    preds_file = results["preds_file"]
    summary_file = results["summary_file"]
    
    # Check predictions CSV file
    assert preds_file.exists(), f"Predictions file not found: {preds_file}"
    assert preds_file.stat().st_size > 0, f"Predictions file is empty: {preds_file}"
    print(f"✓ Predictions CSV file exists and is non-empty: {preds_file}")
    
    # Check predictions file has data
    preds_df = pd.read_csv(preds_file, index_col=0)
    assert len(preds_df) > 0, "Predictions DataFrame is empty"
    assert len(preds_df.columns) > 0, "Predictions DataFrame has no columns"
    print(f"  Predictions shape: {preds_df.shape}")
    
    # Check summary CSV file
    assert summary_file.exists(), f"Summary file not found: {summary_file}"
    assert summary_file.stat().st_size > 0, f"Summary file is empty: {summary_file}"
    print(f"✓ Summary CSV file exists and is non-empty: {summary_file}")
    
    # Check summary file has required columns
    summary_df = pd.read_csv(summary_file)
    required_summary_cols = ["month", "mean", "p10", "p50", "p90"]
    for col in required_summary_cols:
        assert col in summary_df.columns, f"Summary missing column: {col}"
    assert len(summary_df) > 0, "Summary DataFrame is empty"
    print(f"  Summary shape: {summary_df.shape}")
    print(f"  Summary columns: {list(summary_df.columns)}")
    
    # 4. Check feature importance plot exists
    importance_path = results["importance_path"]
    assert importance_path.exists(), f"Feature importance plot not found: {importance_path}"
    assert importance_path.stat().st_size > 0, f"Feature importance plot is empty: {importance_path}"
    print(f"✓ Feature importance plot exists: {importance_path}")
    
    print("\n" + "=" * 70)
    print("✓ All assertions passed!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    try:
        test_pipeline()
        print("\n✅ Smoke test PASSED")
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Smoke test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Smoke test ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

