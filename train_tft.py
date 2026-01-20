import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from pytorch_lightning import Trainer
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader


def prepare_data_for_tft(df):
    df = df.copy()
    df['DateFrom'] = pd.to_datetime(df['DateFrom'])
    df = df.sort_values('DateFrom')

    # Definești 'group' după coloana Machine
    df['group'] = df['Machine']

    # Creezi un index incremental pe grup
    df['time_idx'] = df.groupby('group').cumcount()

    # Elimină rândurile fără date în targeturi principale
    df = df.dropna(subset=['Cg', 'Cgk'], how='all')

    return df


def get_available_targets(df):
    targets = []
    for param in ['Cg', 'Cgk', 'Cm', 'Cmk']:
        if param in df.columns and df[param].notna().any():
            targets.append(param)
    return targets


def train_model_for_target(df, target_col, max_encoder_length=6, max_prediction_length=3):
    print(f"\nTraining model for target: {target_col}")

    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])

    training_cutoff = df["time_idx"].max() - max_prediction_length

    print("Data points per group:")
    print(df.groupby("group").size())
    print("Max time_idx:", df["time_idx"].max())
    print("Training cutoff:", training_cutoff)
    print("Data sample for main group:")
    print(df[df["group"] == "1130-060"].sort_values("time_idx"))

    # Filtrare grupuri cu suficiente date
    min_required_points = max_encoder_length + max_prediction_length
    group_counts = df.groupby("group").size()
    valid_groups = group_counts[group_counts >= min_required_points].index.tolist()
    df = df[df["group"].isin(valid_groups)]

    if df.empty:
        print(f"⚠️  Nu sunt suficiente date pentru antrenarea modelului '{target_col}'.")
        return

    # Dataset pentru antrenament
    training = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target_col,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        min_encoder_length=1,
        max_prediction_length=max_prediction_length,
        min_prediction_idx=0,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Dataset pentru validare
    validation = TimeSeriesDataSet.from_dataset(training, df[lambda x: x.time_idx > training_cutoff])

    train_dataloader = DataLoader(training, batch_size=64, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(validation, batch_size=64, shuffle=False, num_workers=0)

    # Creezi modelul
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,  # implicit pentru cuantile
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    trainer = Trainer(
        max_epochs=20,
        gradient_clip_val=0.1,
        limit_train_batches=30,  # poate fi modificat sau eliminat pentru antrenare completă
        enable_model_summary=True,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def train_all_models(df):
    df = prepare_data_for_tft(df)
    targets = get_available_targets(df)
    for target in targets:
        train_model_for_target(df, target)


if __name__ == "__main__":
    from database.db import get_engine
    from sqlalchemy import text

    query = """
    SELECT   ID
,[DateFrom]
      ,[Cg]
      ,[Cgk]  
      ,[InsertionDate]
      ,[ValidFor] 
      ,[Machine]  
      ,[USL]
      ,[LSL] 
      ,[CharacteristicProductDescription]
 ,[Path]
 ,p.process_name
  FROM [tef8_capability].[tef8_capability].[capability1]  as c
  left join [tef8_capability.equipment] as e
  on c.ProductID = e.ProductID
  left join [tef8_capability.proces] as p
  on e.id_process = p.id_process
  where [Type] like '%1%' and Machine = '1130-060' and CharacteristicProductDescription like '%ILL_XRAY G11 C41 100 80%' and ID > 6812
  order by DateFrom
    """

    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)

    train_all_models(df)
