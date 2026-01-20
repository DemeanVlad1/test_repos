import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

def prepare_data_for_forecast(df):
    df = df.copy()
    df['DateFrom'] = pd.to_datetime(df['DateFrom'])
    df = df.sort_values('DateFrom')
    df['time_idx'] = (df['DateFrom'] - df['DateFrom'].min()).dt.days
    df['group'] = df['Machine']
    return df

def load_model(model_dir, target_col, training_dataset):
    model_path = os.path.join(model_dir, f"tft_{target_col}.ckpt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found for target {target_col}: {model_path}")

    model = TemporalFusionTransformer.load_from_checkpoint(model_path, dataset=training_dataset)
    model.eval()
    return model

def forecast_target(df, target_col, model_dir="models", max_encoder_length=30, max_prediction_length=12):
    df = prepare_data_for_forecast(df)

    # Reload the same dataset config as training
    training_cutoff = df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[target_col],
        target_normalizer=None,  # will be loaded from model
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)
    val_dataloader = DataLoader(validation, batch_size=64, shuffle=False)

    model = load_model(model_dir, target_col, training)

    predictions = model.predict(val_dataloader)
    
    # Vizualizare (opțional)
    raw_predictions, x = model.predict(val_dataloader, mode="raw", return_x=True)
    for idx in range(min(3, len(x["decoder_target"]))):
        model.plot_prediction(x, raw_predictions, idx=idx)
        plt.show()

    return predictions

if __name__ == "__main__":
    from database.db import get_engine
    from sqlalchemy import text

    query = """
    SELECT   ID, [DateFrom], [Cg], [Cgk], [Cm], [Cmk], [InsertionDate], [ValidFor], [Machine],
             [USL], [LSL], [CharacteristicProductDescription], [Path], p.process_name
      FROM [tef8_capability].[tef8_capability].[capability1] as c
      LEFT JOIN [tef8_capability.equipment] as e ON c.ProductID = e.ProductID
      LEFT JOIN [tef8_capability.proces] as p ON e.id_process = p.id_process
      WHERE [Type] LIKE '%1%' AND Machine = '1130-060' AND CharacteristicProductDescription LIKE '%ILL_XRAY G11 C41 100 80%' AND ID > 6812
      ORDER BY DateFrom
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)

    # Exemplu: prezicem pentru Cg
    preds = forecast_target(df, 'Cg')
    print(preds)
