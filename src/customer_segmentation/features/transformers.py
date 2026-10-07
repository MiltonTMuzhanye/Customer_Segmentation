from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from ..utils.logger import get_logger


class CustomerFeatureScaler:
    """Scale customer clustering features."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.scaler = StandardScaler()

    def fit_transform(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fit scaler and transform features."""

        scaled = self.scaler.fit_transform(features)

        result = pd.DataFrame(
            scaled,
            columns=features.columns,
            index=features.index,
        )

        self.logger.info(
            f"Scaled {len(result):,} customers x "
            f"{len(result.columns)} features"
        )

        return result

    def transform(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform features using fitted scaler."""

        scaled = self.scaler.transform(features)

        return pd.DataFrame(
            scaled,
            columns=features.columns,
            index=features.index,
        )

    def save(self, path: str) -> None:
        """Save fitted scaler."""

        Path(path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(self.scaler, path)

        self.logger.info(
            f"Scaler saved to: {path}"
        )
