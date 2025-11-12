"""Model training and evaluation modules."""

from finmc_tech.models.rf_model import (
    fit_rf,
    evaluate_rf,
    save_model,
    load_model,
    plot_feature_importance,
    train_rf_model,  # Backward compatibility
    evaluate_rf_model,  # Backward compatibility
)

# LSTM imports (optional, may raise ImportError if tensorflow not installed)
try:
    from finmc_tech.models.lstm_model import (
        build_lstm,
        to_sequences,
        fit_lstm,
        predict_lstm,
        train_lstm_model,  # Backward compatibility
        prepare_lstm_data,  # Backward compatibility
    )
    __all__ = [
        "fit_rf",
        "evaluate_rf",
        "save_model",
        "load_model",
        "plot_feature_importance",
        "train_rf_model",
        "evaluate_rf_model",
        "build_lstm",
        "to_sequences",
        "fit_lstm",
        "predict_lstm",
        "train_lstm_model",
        "prepare_lstm_data",
    ]
except ImportError:
    __all__ = [
        "fit_rf",
        "evaluate_rf",
        "save_model",
        "load_model",
        "plot_feature_importance",
        "train_rf_model",
        "evaluate_rf_model",
    ]

