import base64
from io import BytesIO
from typing import Dict, List, Optional

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

    def generate_waterfall_plot(self, X, output_format: str = "base64") -> Optional[str]:
        try:
            shap_values = self.explainer.shap_values(X)
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value,
                    data=X.iloc[0].values,
                    feature_names=self.feature_names,
                ),
                show=False,
            )
            if output_format == "base64":
                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                image_b64 = base64.b64encode(buf.read()).decode()
                plt.close()
                return image_b64
            return None
        except Exception as e:
            logger.error(f"Error generating SHAP waterfall: {e}")
            return None

