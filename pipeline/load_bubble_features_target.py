from typing import List, Optional, Union
import datetime
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .constant import features_name
# from utils import set_seed
from transformers import set_seed


# Suppress all warnings
# warnings.filterwarnings("ignore")

# ignore from terminal
#python -W ignore test_load_bubble_features_target.py 


class LoadBubbleFeaturesTarget:

    def __init__(
        self,
        news_path: str,
        features_economic_path: str,
        target_path: str,
        sentiment_window_size: int,
        bubble_direction_window_size: int,
        bubble_direction_threshold: int = Optional[None],
        include_current_in_rolling_bubble_direction: bool = False,
        start_date: Union[str, datetime.date] = None,
        end_date: Union[str, datetime.date] = None,
        test_size: int = 0.2,
        seed: int = 42,
        ignore_warnings: bool = True,
    ):
        if seed:
            set_seed(seed)

        if ignore_warnings:
            warnings.filterwarnings("ignore")
        news_df = pd.read_csv(news_path)
        news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
        news_df["date"] = news_df["date"].dt.strftime("%Y-%m-%d")
        # features
        features_economic_df = pd.read_csv(features_economic_path)
        features_economic_df["Date"] = pd.to_datetime(
            features_economic_df["Date"], errors="coerce"
        )
        features_economic_df["Date"] = features_economic_df["Date"].dt.strftime(
            "%Y-%m-%d"
        )
        # target
        target = pd.read_csv(target_path)
        target["Date"] = pd.to_datetime(target["Date"], errors="coerce")
        target["Date"] = target["Date"].dt.strftime("%Y-%m-%d")

        news_df["rolling_average_sentiment"] = (
            news_df["average_sentiment"].rolling(window=sentiment_window_size).mean()
        )
        news_df["rolling_total_sentiment"] = (
            news_df["total_sentiment"].rolling(window=sentiment_window_size).sum()
        )
        news_df["rolling_average_sentiment"].fillna(
            value=np.mean(news_df["rolling_average_sentiment"]), inplace=True
        )
        news_df["rolling_total_sentiment"].fillna(
            value=np.mean(news_df["rolling_total_sentiment"]), inplace=True
        )
        # mearge features
        features = pd.merge(
            left=features_economic_df,
            right=news_df,
            left_on="Date",
            right_on="date",
            how="left",
        )
        features.drop_duplicates(subset=["Date"], inplace=True)
        target["date_mod"] = target["Date"].apply(func=self.target_date_mod)
        # target["date_mod"] = pd.to_datetime(target["date_mod"], errors="coerce")
        # target and feature
        target_feature = pd.merge(
            left=target, right=features, left_on="date_mod", right_on="Date", how="left"
        )

        # calculate bubble up and down
        for col in ["interpolate"]:
            target_feature[f"{col}_rolling_mean_excl_current"] = (
                self.rolling_mean_excluding_current(
                    target_feature[col], bubble_direction_window_size
                )
            )
        target_feature[f"interpolate_rolling_mean_incl_current"] = (
            target_feature["interpolate"]
            .rolling(window=bubble_direction_window_size)
            .mean()
        )

        for col in target_feature.columns:
            if (
                "interpolate_rolling_mean_excl_current" in col
                or "interpolate_rolling_mean_incl_current" in col
            ):
                mean_value = target_feature[col].mean()
                target_feature[col].fillna(mean_value, inplace=True)

        if include_current_in_rolling_bubble_direction:
            if bubble_direction_threshold:
                target_feature["is_bubble_up"] = target_feature.apply(
                    lambda row: self.bubble_up_threshold(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_incl_current"],
                        bubble_direction_threshold,
                    ),
                    axis=1,
                )
                target_feature["is_bubble_down"] = target_feature.apply(
                    lambda row: self.bubble_down_threshold(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_incl_current"],
                        bubble_direction_threshold,
                    ),
                    axis=1,
                )
            else:
                target_feature["is_bubble_up"] = target_feature.apply(
                    lambda row: self.bubble_up(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_incl_current"],
                        row["interpolate"],
                    ),
                    axis=1,
                )
                target_feature["is_bubble_down"] = target_feature.apply(
                    lambda row: self.bubble_down(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_incl_current"],
                        row["interpolate"],
                    ),
                    axis=1,
                )
        else:
            if bubble_direction_threshold:
                target_feature["is_bubble_up"] = target_feature.apply(
                    lambda row: self.bubble_up_threshold(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_excl_current"],
                        bubble_direction_threshold,
                    ),
                    axis=1,
                )
                target_feature["is_bubble_down"] = target_feature.apply(
                    lambda row: self.bubble_down_threshold(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_excl_current"],
                        bubble_direction_threshold,
                    ),
                    axis=1,
                )
            else:
                target_feature["is_bubble_up"] = target_feature.apply(
                    lambda row: self.bubble_up(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_excl_current"],
                        row["interpolate"],
                    ),
                    axis=1,
                )
                target_feature["is_bubble_down"] = target_feature.apply(
                    lambda row: self.bubble_down(
                        row["is_bubble"],
                        row["interpolate_rolling_mean_excl_current"],
                        row["interpolate"],
                    ),
                    axis=1,
                )
        target_col = [
            # "date_mod",
            # "interpolate",
            # "interpolate_rolling_mean_excl_current",
            # "interpolate_rolling_mean_incl_current",
            "is_bubble",
            "not_bubble",
            "is_bubble_up",
            "is_bubble_down",
        ]

        features_col = (
            # ["date_mod"]
            list(features_name.keys())
            + [
                # "total_sentiment",
                "average_sentiment",
                # "rolling_total_sentiment",
                # "rolling_average_sentiment",
            ]
        )

        all_columns = ["date_mod"] + target_col[1:] + features_col[1:]

        # filter by date
        target_date = target_feature["date_mod"].to_list()
        if not start_date:
            start_date = target_date[0]
        if not end_date:
            end_date = target_date[-1]
        filtered = target_feature[
            (target_feature["date_mod"] >= start_date)
            & (target_feature["date_mod"] <= end_date)
        ]

        columns_to_fill = [
            "total_sentiment",
            "average_sentiment",
            "rolling_total_sentiment",
            "rolling_average_sentiment",
        ] + list(features_name.keys())
        # Fill NaN values with the median of each column
        for col in columns_to_fill:
            median_value = filtered[col].median()
            filtered[col].fillna(median_value, inplace=True)

        features_new = filtered[features_col]
        features_new = features_new.dropna()
        target_new = filtered[target_col]
        target_new = target_new.dropna()
        features_new.rename(columns=features_name, inplace=True)

        # checking len
        if len(target_new) != len(features_new):
            raise ValueError(
                "Data Contains missing values or number of rows does not match"
            )

        # time lag features
        target_new = target_new.iloc[1:]
        features_new = features_new.iloc[:-1]
        features_new.dropna(inplace=True)
        # features_new.dropna(inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            features_new,
            target_new,
            test_size=test_size,
            random_state=seed,
            stratify=target_new,
        )
        self.target = target_new
        self.features = features_new
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def target_date_mod(self, date):
        date_split = date.split("-")
        year, month, day = (
            np.int16(date_split[0]),
            np.int16(date_split[1]),
            np.int16(date_split[2]),
        )
        if day > 15:
            day = 1
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1
        else:
            day = 15
        if month <= 9:
            month = f"0{month}"
        if day <= 9:
            day = f"0{day}"

        date = f"{year}-{month}-{day}"
        return date

    # creating bubble up and bubble down
    def bubble_up(self, is_bubble, x_rolling, x_current):
        if is_bubble == 1 and x_current > x_rolling:
            return 1
        return 0

    def bubble_down(self, is_bubble, x_rolling, x_current):
        if is_bubble == 1 and x_current <= x_rolling:
            return 1
        return 0

    # creating bubble up and bubble down: for threshold
    def bubble_up_threshold(self, is_bubble, x_rolling, threshold):
        if is_bubble == 1 and x_rolling > threshold:
            return 1
        return 0

    def bubble_down_threshold(self, is_bubble, x_rolling, threshold):
        if is_bubble == 1 and x_rolling <= threshold:
            return 1
        return 0

    def rolling_mean_excluding_current(self, series, window_size):
        result = []
        for i in range(len(series)):
            start = max(0, i - window_size)
            end = i
            window = list(series[start:end])
            result.append(np.mean(window))
        return result
