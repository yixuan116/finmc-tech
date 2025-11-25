"""Command-line interface for macro-driven NVDA simulation."""

import argparse
import json
from pathlib import Path
from finmc_tech.config import Settings, cfg, logger
from finmc_tech.sim.run_simulation import pipeline
from finmc_tech.viz.plots import (
    plot_pred_vs_actual,
    plot_sim_distribution,
    plot_rolling_corr,
)
import pandas as pd
import numpy as np


def train_rf_cmd(args):
    """Train RandomForest model."""
    logger.info("=" * 70)
    logger.info("Training RandomForest Model")
    logger.info("=" * 70)
    
    config = Settings(
        TICKER=args.ticker,
        START_DATE=args.start_date,
        END_DATE=args.end_date or cfg.END_DATE,
        CACHE_DIR=args.cache_dir,
        RESULTS_DIR=args.output_dir,
        RANDOM_STATE=cfg.RANDOM_STATE,
    )
    
    # Run pipeline (training only, no simulation)
    results = pipeline(
        config=config,
        H=1,  # Minimal horizon for training
        n_paths=1,  # Minimal paths for training
        shock="base",
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("Training Complete!")
    logger.info("=" * 70)
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"Feature importance plot: {results['importance_path']}")
    logger.info(f"Test R²: {results['metrics']['R2']:.4f}")
    logger.info(f"Test RMSE: {results['metrics']['RMSE']:.4f}")
    logger.info(f"Test MAE: {results['metrics']['MAE']:.4f}")


def simulate_cmd(args):
    """Run Monte Carlo simulation."""
    logger.info("=" * 70)
    logger.info("Running Monte Carlo Simulation")
    logger.info("=" * 70)
    
    config = Settings(
        TICKER=args.ticker,
        START_DATE=args.start_date,
        END_DATE=args.end_date or cfg.END_DATE,
        CACHE_DIR=args.cache_dir,
        RESULTS_DIR=args.output_dir,
        RANDOM_STATE=cfg.RANDOM_STATE,
    )
    
    # Run full pipeline with simulation
    results = pipeline(
        config=config,
        H=args.h,
        n_paths=args.n,
        shock=args.shock,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("Simulation Complete!")
    logger.info("=" * 70)
    logger.info(f"Predictions saved to: {results['preds_file']}")
    logger.info(f"Summary saved to: {results['summary_file']}")


def plots_cmd(args):
    """Generate plots from results."""
    logger.info("=" * 70)
    logger.info("Generating Plots")
    logger.info("=" * 70)
    
    config = Settings(
        TICKER=args.ticker,
        RESULTS_DIR=args.output_dir,
    )
    
    results_dir = Path(config.RESULTS_DIR)
    
    plots_generated = []
    
    if args.which in ["all", "predictions"]:
        # Load test results if available
        results_file = results_dir / f"{config.TICKER}_results.json"
        if results_file.exists():
            with open(results_file, "r") as f:
                results = json.load(f)
            
            if "test_metrics" in results and "predictions" in results["test_metrics"]:
                y_test = np.array(results["test_metrics"]["predictions"]["y_test"])
                y_pred = np.array(results["test_metrics"]["predictions"]["y_pred"])
                dates = pd.date_range(start="2020-01-01", periods=len(y_test), freq="M")
                
                outpath = results_dir / f"{config.TICKER}_pred_vs_actual.png"
                plot_pred_vs_actual(dates, y_test, y_pred, outpath)
                plots_generated.append(outpath)
    
    if args.which in ["all", "simulation"]:
        # Load simulation results
        preds_file = results_dir / f"macro_mc_predictions_{args.shock}_H{args.h}_n{args.n}.csv"
        if not preds_file.exists():
            # Try to find any prediction file
            preds_files = list(results_dir.glob("macro_mc_predictions_*.csv"))
            if preds_files:
                preds_file = preds_files[0]
        
        if preds_file.exists():
            preds_df = pd.read_csv(preds_file, index_col=0)
            outpath = results_dir / f"{config.TICKER}_sim_distribution.png"
            plot_sim_distribution(preds_df, outpath)
            plots_generated.append(outpath)
    
    if args.which in ["all", "correlation"]:
        # Load aligned data for rolling correlation
        from finmc_tech.data.fetch_macro import fetch_macro
        from finmc_tech.data.fetch_firm import fetch_firm_data
        from finmc_tech.data.align import align_macro_firm
        
        macro_df = fetch_macro(config.START_DATE, config.END_DATE, config.cache_dir)
        firm_df = fetch_firm_data(config.TICKER, config.START_DATE, config.END_DATE, config.cache_dir)
        panel = align_macro_firm(macro_df, firm_df)
        
        outpath = results_dir / f"{config.TICKER}_rolling_corr.png"
        plot_rolling_corr(panel, outpath)
        plots_generated.append(outpath)
    
    logger.info("\n" + "=" * 70)
    logger.info(f"Generated {len(plots_generated)} plots:")
    for plot_path in plots_generated:
        logger.info(f"  ✓ {plot_path}")
    logger.info("=" * 70)


def step5_cmd(args):
    """Extract key drivers from champion RF model (Step 5)."""
    from finmc_tech.step5_key_drivers import run_step5
    
    run_step5(
        data_path=getattr(args, 'data_path', "data/processed/nvda_features_extended.csv"),
        target_column=getattr(args, 'target_column', "return_next_month"),
        model_path=getattr(args, 'model_path', "models/champion_model.pkl"),
        scaler_path=getattr(args, 'scaler_path', "models/feature_scaler.pkl"),
        output_dir=getattr(args, 'output_dir', "results/step5"),
        top_n=getattr(args, 'top_n', 5),
        rolling=getattr(args, 'rolling', False),
    )


def simulate_scenarios_cmd(args):
    """Run Step 7: Scenario-based Monte Carlo forecasting."""
    logger.info("=" * 70)
    logger.info("Running Step 7: Scenario-Based Monte Carlo Forecasting")
    logger.info("=" * 70)
    
    from finmc_tech.simulation.scenario_mc import run_scenario_forecast
    
    results = run_scenario_forecast(
        ticker=args.ticker,
        horizon_months=args.h,
        n_sims=args.n,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        output_dir=args.output_dir,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("Step 7 Analysis Complete!")
    logger.info("=" * 70)
    logger.info(f"Forecast table saved to: {args.output_dir}/scenario_forecast_table.csv")
    logger.info(f"Fan charts saved to: {args.output_dir}/fan_chart_*.png")
    logger.info(f"Distribution plots saved to: {args.output_dir}/distribution_shift_*.png")


def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(
        description="Macro-driven NVDA return simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--ticker", default="NVDA", help="Stock ticker")
    common_parser.add_argument("--start-date", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    common_parser.add_argument("--end-date", default=None, help="End date (YYYY-MM-DD)")
    common_parser.add_argument("--output-dir", default="results", help="Output directory")
    common_parser.add_argument("--cache-dir", default="data_cache", help="Cache directory")
    
    # train-rf subcommand
    train_parser = subparsers.add_parser(
        "train-rf",
        parents=[common_parser],
        help="Train RandomForest model",
    )
    train_parser.set_defaults(func=train_rf_cmd)
    
    # simulate subcommand
    sim_parser = subparsers.add_parser(
        "simulate",
        parents=[common_parser],
        help="Run Monte Carlo simulation",
    )
    sim_parser.add_argument("--shock", choices=["base", "stress"], default="base",
                           help="Shock type: base or stress")
    sim_parser.add_argument("--h", type=int, default=24,
                           help="Horizon in months")
    sim_parser.add_argument("--n", type=int, default=200,
                           help="Number of paths")
    sim_parser.set_defaults(func=simulate_cmd)
    
    # plots subcommand
    plots_parser = subparsers.add_parser(
        "plots",
        parents=[common_parser],
        help="Generate plots from results",
    )
    plots_parser.add_argument("--which", choices=["all", "predictions", "simulation", "correlation"],
                             default="all", help="Which plots to generate")
    plots_parser.add_argument("--shock", default="base", help="Shock type for simulation plots")
    plots_parser.add_argument("--h", type=int, default=24, help="Horizon for simulation plots")
    plots_parser.add_argument("--n", type=int, default=200, help="Number of paths for simulation plots")
    plots_parser.set_defaults(func=plots_cmd)
    
    # step5 subcommand
    step5_parser = subparsers.add_parser(
        "step5",
        help="Extract key drivers from champion RF model (Step 5)",
    )
    step5_parser.add_argument("--data-path", default="data/processed/nvda_features_extended.csv",
                             help="Path to feature CSV")
    step5_parser.add_argument("--target-column", default="return_next_month",
                             help="Target column name")
    step5_parser.add_argument("--model-path", default="models/champion_model.pkl",
                             help="Path to champion model")
    step5_parser.add_argument("--scaler-path", default="models/feature_scaler.pkl",
                             help="Path to feature scaler")
    step5_parser.add_argument("--output-dir", default="results/step5",
                             help="Output directory")
    step5_parser.add_argument("--top-n", type=int, default=5,
                             help="Number of top drivers for PDP/ICE plots")
    step5_parser.add_argument("--rolling", action="store_true",
                             help="Generate rolling importance plot")
    step5_parser.set_defaults(func=step5_cmd)
    
    # simulate-scenarios subcommand
    sim_scenarios_parser = subparsers.add_parser(
        "simulate-scenarios",
        parents=[common_parser],
        help="Run Step 7: Scenario-based Monte Carlo forecasting",
    )
    sim_scenarios_parser.add_argument("--h", type=int, default=12,
                                     help="Horizon in months (default: 12)")
    sim_scenarios_parser.add_argument("--n", type=int, default=5000,
                                     help="Number of simulations (default: 5000)")
    sim_scenarios_parser.add_argument("--model-path", default=None,
                                     help="Path to champion model (default: models/champion_model.pkl)")
    sim_scenarios_parser.add_argument("--scaler-path", default=None,
                                     help="Path to feature scaler (default: models/feature_scaler.pkl)")
    sim_scenarios_parser.set_defaults(func=simulate_scenarios_cmd)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
