import base64
from io import BytesIO
from typing import Dict, List, Optional, Any

import numpy as np

import matplotlib.pyplot as plt
import shap

from src.utils.logging import get_logger


logger = get_logger(__name__)


class PredictionExplainer:
    """SHAP-based explainability for LightGBM models."""

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        # TreeExplainer works with LightGBM boosters
        self.explainer = shap.TreeExplainer(model)

    def get_feature_importance(self, top_n: int = 5) -> Dict[str, float]:
        importance = self.model.feature_importance(importance_type="gain")
        mapping = dict(zip(self.feature_names, importance))
        return dict(sorted(mapping.items(), key=lambda x: x[1], reverse=True)[:top_n])

    def compute_waterfall_data(self, X) -> Optional[Dict[str, Any]]:
        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_vector = np.array(shap_values[0])[0]
            else:
                shap_vector = np.array(shap_values)[0]

            base_value_raw = self.explainer.expected_value
            base_value = float(np.array(base_value_raw).flatten()[0])
            feature_values = X.iloc[0].values

            features = []
            for name, value, contribution in zip(self.feature_names, feature_values, shap_vector):
                features.append(
                    {
                        "feature": name,
                        "value": float(value),
                        "contribution": float(contribution),
                    }
                )

            output_value = float(base_value + float(np.sum(shap_vector)))

            return {
                "base_value": base_value,
                "output_value": output_value,
                "features": features,
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error computing SHAP waterfall data: {e}")
            return None

    def generate_waterfall(self, X, include_plot: bool = True) -> Dict[str, Optional[str]]:
        payload = {"plot_base64": None, "data": self.compute_waterfall_data(X)}

        if not include_plot:
            return payload

        try:
            shap_values = self.explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_vector = shap_values[0]
            else:
                shap_vector = shap_values

            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=np.array(shap_vector)[0],
                    base_values=self.explainer.expected_value,
                    data=X.iloc[0].values,
                    feature_names=self.feature_names,
                ),
                show=False,
            )
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            payload["plot_base64"] = base64.b64encode(buf.read()).decode()
            plt.close()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error generating SHAP waterfall plot: {e}")
        return payload

    def generate_waterfall_plot(self, X, output_format: str = "base64") -> Optional[str]:
        try:
            result = self.generate_waterfall(X, include_plot=(output_format == "base64"))
            return result.get("plot_base64")
        except Exception as e:
            logger.error(f"Error generating SHAP waterfall: {e}")
            return None
