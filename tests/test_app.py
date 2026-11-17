import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app import app, pipeline

client = TestClient(app)

@pytest.fixture(scope="module")
def genuine_sample():
    """Retrieve an authentic test session sample for s002."""
    df = pd.read_csv("data/DSL-StrongPasswordData.csv")
    sample = df[(df["subject"] == "s002") & (df["sessionIndex"].isin([5, 6, 7, 8]))].iloc[0]
    return {col: float(sample[col]) for col in pipeline.feature_names}

@pytest.fixture(scope="module")
def impostor_sample():
    """Retrieve an authentic test session sample from impostor s003."""
    df = pd.read_csv("data/DSL-StrongPasswordData.csv")
    sample = df[(df["subject"] == "s003") & (df["sessionIndex"].isin([5, 6, 7, 8]))].iloc[0]
    return {col: float(sample[col]) for col in pipeline.feature_names}

def test_health_endpoint():
    """Stage 5 Test 1: Verify health check route returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["enrolled_subject"] == "s002"
    assert "operating_threshold" in data

def test_ui_index_page():
    """Stage 5 Test 2: Verify root UI loads and serves HTML successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Keystroke Dynamics Biometric Authentication" in response.text
    assert "Authenticate Biometric Sample" in response.text

def test_domain_logic_genuine_vs_impostor(genuine_sample, impostor_sample):
    """
    Stage 5 Test 3 (Domain-Logic Test):
    An enrolled genuine typist MUST score significantly higher than an impostor.
    Equivalent to 9.0 CGPA student scoring higher than 5.5 CGPA student in reference project.
    """
    res_gen = client.post("/predict", json=genuine_sample)
    assert res_gen.status_code == 200
    gen_data = res_gen.json()

    res_imp = client.post("/predict", json=impostor_sample)
    assert res_imp.status_code == 200
    imp_data = res_imp.json()

    # Domain assertions:
    # 1. Genuine match score must exceed impostor score
    assert gen_data["match_score"] > imp_data["match_score"], (
        f"Domain logic violation: Genuine score ({gen_data['match_score']}) "
        f"not higher than impostor score ({imp_data['match_score']})"
    )
    # 2. Genuine sample must be authenticated
    assert gen_data["authenticated"] is True
    assert gen_data["prediction"] == 1
    # 3. Impostor sample must be rejected
    assert imp_data["authenticated"] is False
    assert imp_data["prediction"] == 0

def test_invalid_input_missing_field_returns_422(genuine_sample):
    """Stage 5 Test 4: Missing feature in payload must return HTTP 422."""
    bad_payload = genuine_sample.copy()
    del bad_payload["H.period"]  # Omit required feature
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422, f"Expected 422 Unprocessable Entity, got {response.status_code}"

def test_boundary_validation_out_of_range_returns_422(genuine_sample):
    """Stage 5 Test 5: Out-of-bounds feature value must return HTTP 422."""
    bad_payload = genuine_sample.copy()
    bad_payload["H.period"] = 999.0  # Physically impossible hold time (max 3.0s)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422

    # Negative hold time is physically impossible
    bad_payload2 = genuine_sample.copy()
    bad_payload2["H.Return"] = -0.1
    response2 = client.post("/predict", json=bad_payload2)
    assert response2.status_code == 422

def test_negative_ud_rollover_accepted(genuine_sample):
    """
    Stage 5 Test 6: Verify that authentic negative UD timings (key rollover)
    are strictly accepted and not falsely rejected by contract validation.
    """
    valid_rollover_payload = genuine_sample.copy()
    valid_rollover_payload["UD.i.e"] = -0.08  # Valid key rollover
    response = client.post("/predict", json=valid_rollover_payload)
    assert response.status_code == 200
