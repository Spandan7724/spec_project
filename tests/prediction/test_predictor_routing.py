from src.prediction.models import PredictionRequest
from src.prediction.predictor import MLPredictor


def test_split_horizons_routing():
    predictor = MLPredictor()
    req = PredictionRequest(
        currency_pair="USD/EUR",
        horizons=[1, 7, 30],
        intraday_horizons_hours=[1, 4, 24],
    )
    daily, intra = predictor._split_horizons(req)
    assert daily == [1, 7, 30]
    assert intra == [1, 4, 24]

