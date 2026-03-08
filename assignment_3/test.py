import os
import sys
import time
import subprocess
import joblib
import requests
import pytest

from score import score

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.joblib')
model = joblib.load(MODEL_PATH)
THRESHOLD = 0.5


def test_score_smoke():
    """smoke test"""
    prediction, propensity = score("hello", model, THRESHOLD)


def test_score_output_types():
    """check output types"""
    prediction, propensity = score("hello", model, THRESHOLD)
    assert isinstance(prediction, (bool,))
    assert isinstance(propensity, float)


def test_score_prediction_binary():
    """prediction should be True or False"""
    prediction, propensity = score("hello", model, THRESHOLD)
    assert prediction in (True, False)


def test_score_propensity_range():
    """propensity between 0 and 1"""
    prediction, propensity = score("hello", model, THRESHOLD)
    assert 0.0 <= propensity <= 1.0

def test_score_threshold_zero():
    """threshold=0 means everything is spam"""
    prediction, _ = score("hello", model, 0.0)
    assert prediction == True


def test_score_threshold_one():
    """threshold=1 means nothing is spam"""
    prediction, propensity = score("hello", model, 1.0)
    # propensity is strictly < 1 for any real input, so prediction should be False
    assert prediction == False

def test_score_spam_input():
    """obvious spam should be caught"""
    spam_text = "WINNER! You have been selected to receive a $1000 prize. Call now to claim your FREE reward!"
    prediction, propensity = score(spam_text, model, THRESHOLD)
    assert prediction == True


def test_score_ham_input():
    """normal text should not be spam"""
    ham_text = "Hey, are we still meeting for lunch tomorrow at noon?"
    prediction, propensity = score(ham_text, model, THRESHOLD)
    assert prediction == False


# Flask integration test

def test_flask():
    """start flask app, hit /score, then shut down"""
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')

    # Launch Flask app as a subprocess
    proc = subprocess.Popen(
        [sys.executable, app_path],
        cwd=os.path.dirname(__file__),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for the server to start
        url = 'http://127.0.0.1:5000/score'
        for _ in range(30):
            try:
                resp = requests.post(url, json={'text': 'test'}, timeout=2)
                if resp.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)
        else:
            pytest.fail("Flask app did not start in time")

        # Test with a spam message
        resp = requests.post(url, json={'text': 'FREE money! Call now to claim your prize!'}, timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert 'prediction' in data
        assert 'propensity' in data
        assert isinstance(data['prediction'], bool)
        assert isinstance(data['propensity'], float)
        assert 0.0 <= data['propensity'] <= 1.0
    finally:
        proc.terminate()
        proc.wait(timeout=5)
