import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from olist_forecast import recommend_capacity  # noqa: E402
from olist_forecast.api import create_app  # noqa: E402


@pytest.fixture()
def client(series):
    return TestClient(create_app(lambda: series))


def test_health_does_not_need_the_data(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "data_loaded": False}


def test_recommendation_matches_the_package(client, series):
    response = client.post("/recommend", json={"cu": 4, "co": 1})
    assert response.status_code == 200
    body = response.json()
    expected = recommend_capacity(series, 4, 1)
    assert body["as_of"] == series.index.max().date().isoformat()
    assert body["critical_ratio"] == 0.8
    assert len(body["items"]) == len(expected)
    first = body["items"][0]
    assert first["category"] == expected.loc[0, "category"]
    assert first["capacity"] == pytest.approx(expected.loc[0, "capacity"], abs=0.01)
    assert client.get("/health").json()["data_loaded"] is True


def test_category_filter(client):
    body = client.post("/recommend", json={"cu": 4, "co": 1, "categories": ["a"]}).json()
    assert {item["category"] for item in body["items"]} == {"a"}
    assert len(body["items"]) == 7


def test_unknown_category_is_rejected_with_the_available_ones(client):
    response = client.post("/recommend", json={"cu": 4, "co": 1, "categories": ["zzz"]})
    assert response.status_code == 422
    assert "Available" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"cu": 0, "co": 1},
        {"cu": 1, "co": -1},
        {"cu": 1000, "co": 1},  # critical ratio above 0.99
        {"cu": 1},
        {"cu": "four", "co": 1},
    ],
)
def test_invalid_costs_are_rejected(client, payload):
    assert client.post("/recommend", json=payload).status_code == 422


def test_scaled_widening_option_is_accepted(client):
    response = client.post("/recommend", json={"cu": 4, "co": 1, "scaled_widening": True})
    assert response.status_code == 200


def test_repeated_requests_use_the_cache(series):
    calls = []

    def load():
        calls.append(1)
        return series

    client = TestClient(create_app(load))
    first = client.post("/recommend", json={"cu": 4, "co": 1}).json()
    second = client.post("/recommend", json={"cu": 8, "co": 2}).json()
    assert first == second
    assert len(calls) == 1


def test_service_answers_503_when_the_database_is_not_ready():
    def broken():
        raise RuntimeError("relation orders does not exist")

    response = TestClient(create_app(broken)).post("/recommend", json={"cu": 4, "co": 1})
    assert response.status_code == 503
    assert "could not be loaded" in response.json()["detail"]


def test_service_answers_503_when_the_series_is_empty():
    response = TestClient(create_app(lambda: pd.DataFrame())).post("/recommend", json={"cu": 4, "co": 1})
    assert response.status_code == 503


def test_capacity_units_is_a_whole_number_just_above_the_capacity(client):
    items = client.post("/recommend", json={"cu": 4, "co": 1}).json()["items"]
    for item in items:
        assert isinstance(item["capacity_units"], int)
        assert 0 <= item["capacity_units"] - item["capacity"] < 1.005


def test_day_filter_returns_only_that_day(client, series):
    day = (series.index.max() + pd.Timedelta(days=3)).date().isoformat()
    body = client.post("/recommend", json={"cu": 4, "co": 1, "day": day}).json()
    assert {item["day"] for item in body["items"]} == {day}
    assert len(body["items"]) == series.shape[1]


def test_day_outside_the_forecast_is_rejected(client, series):
    day = (series.index.max() + pd.Timedelta(days=8)).date().isoformat()
    response = client.post("/recommend", json={"cu": 4, "co": 1, "day": day})
    assert response.status_code == 422
    assert "forecast days" in response.json()["detail"]


def test_as_of_moves_the_forecast_and_shows_what_really_happened(client, series):
    as_of = series.index[-30]
    body = client.post("/recommend", json={"cu": 4, "co": 1, "as_of": as_of.date().isoformat()}).json()
    assert body["as_of"] == as_of.date().isoformat()
    first_day = (as_of + pd.Timedelta(days=1)).date().isoformat()
    assert min(item["day"] for item in body["items"]) == first_day
    item = next(i for i in body["items"] if i["category"] == "a" and i["day"] == first_day)
    assert item["actual"] == series.loc[as_of + pd.Timedelta(days=1), "a"]


def test_actual_is_missing_for_days_not_yet_observed(client):
    items = client.post("/recommend", json={"cu": 4, "co": 1}).json()["items"]
    assert all(item["actual"] is None for item in items)


def test_as_of_equal_to_the_default_gives_the_same_answer(client, series):
    default = client.post("/recommend", json={"cu": 4, "co": 1}).json()
    explicit = client.post(
        "/recommend", json={"cu": 4, "co": 1, "as_of": series.index.max().date().isoformat()}
    ).json()
    assert default == explicit


@pytest.mark.parametrize("offset", [-1000, 1])
def test_as_of_outside_the_data_is_rejected(client, series, offset):
    day = (series.index.max() + pd.Timedelta(days=offset)).date().isoformat()
    if offset < 0:
        day = (series.index.min() + pd.Timedelta(days=offset)).date().isoformat()
    response = client.post("/recommend", json={"cu": 4, "co": 1, "as_of": day})
    assert response.status_code == 422


def test_as_of_too_early_to_train_is_rejected(client, series):
    day = (series.index.min() + pd.Timedelta(days=40)).date().isoformat()
    assert client.post("/recommend", json={"cu": 4, "co": 1, "as_of": day}).status_code == 422
