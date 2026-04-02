"""Unit tests for mechanics_engine module."""

import sys
import os
import tempfile
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.mechanics_engine import MechanicsEngine, MechanicResult


@pytest.fixture
def sample_config(tmp_path):
    """Create a temporary YAML config for testing."""
    config = {
        "mechanics": {
            "knee_angle": {
                "display_name": "Knee Bend",
                "good_range": [95, 125],
                "warning_range": [80, 140],
                "feedback": {
                    "good": "Strong knee bend",
                    "warning": "Needs more bend",
                    "poor": "Insufficient bend",
                },
            },
            "trunk_alignment": {
                "display_name": "Trunk Balance",
                "good_range": [-8, 8],
                "warning_range": [-15, 15],
                "feedback": {
                    "good": "Balanced",
                    "warning": "Slight lean",
                    "poor": "Major imbalance",
                },
            },
        },
        "min_visibility": 0.5,
        "analyze_sides": "both",
    }

    config_path = tmp_path / "test_mechanics.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


class TestMechanicsEngine:
    """Tests for the mechanics evaluation engine."""

    def test_good_rating(self, sample_config):
        """Angle in good range gets 'good' rating."""
        engine = MechanicsEngine(config_path=sample_config)
        angles = {"left_knee_angle": 110, "right_knee_angle": 105, "trunk_alignment": 2}
        results = engine.evaluate(angles)

        knee_results = [r for r in results if r.name == "knee_angle"]
        assert len(knee_results) == 2
        assert all(r.rating == "good" for r in knee_results)

    def test_warning_rating(self, sample_config):
        """Angle in warning range gets 'warning' rating."""
        engine = MechanicsEngine(config_path=sample_config)
        angles = {"left_knee_angle": 85, "right_knee_angle": 135, "trunk_alignment": 0}
        results = engine.evaluate(angles)

        knee_results = [r for r in results if r.name == "knee_angle"]
        assert all(r.rating == "warning" for r in knee_results)

    def test_poor_rating(self, sample_config):
        """Angle outside warning range gets 'poor' rating."""
        engine = MechanicsEngine(config_path=sample_config)
        angles = {"left_knee_angle": 170, "right_knee_angle": 60, "trunk_alignment": 0}
        results = engine.evaluate(angles)

        knee_results = [r for r in results if r.name == "knee_angle"]
        assert all(r.rating == "poor" for r in knee_results)

    def test_trunk_alignment(self, sample_config):
        """Trunk alignment (bilateral) classified correctly."""
        engine = MechanicsEngine(config_path=sample_config)

        # Good
        results = engine.evaluate({"trunk_alignment": 3})
        trunk = [r for r in results if r.name == "trunk_alignment"]
        assert len(trunk) == 1
        assert trunk[0].rating == "good"

        # Warning
        results = engine.evaluate({"trunk_alignment": 12})
        trunk = [r for r in results if r.name == "trunk_alignment"]
        assert trunk[0].rating == "warning"

        # Poor
        results = engine.evaluate({"trunk_alignment": 25})
        trunk = [r for r in results if r.name == "trunk_alignment"]
        assert trunk[0].rating == "poor"

    def test_none_values_skipped(self, sample_config):
        """None angle values are gracefully skipped."""
        engine = MechanicsEngine(config_path=sample_config)
        angles = {"left_knee_angle": None, "right_knee_angle": 110, "trunk_alignment": 0}
        results = engine.evaluate(angles)

        knee_results = [r for r in results if r.name == "knee_angle"]
        assert len(knee_results) == 1  # Only right side

    def test_feedback_text(self, sample_config):
        """Correct feedback text returned for each rating."""
        engine = MechanicsEngine(config_path=sample_config)

        results = engine.evaluate({"left_knee_angle": 110, "right_knee_angle": 110, "trunk_alignment": 0})
        knee_good = [r for r in results if r.name == "knee_angle"][0]
        assert knee_good.feedback == "Strong knee bend"

    def test_boundary_values(self, sample_config):
        """Values at exact boundaries are inclusive."""
        engine = MechanicsEngine(config_path=sample_config)

        # At good_range boundary
        results = engine.evaluate({"left_knee_angle": 95, "right_knee_angle": 125, "trunk_alignment": 0})
        knee_results = [r for r in results if r.name == "knee_angle"]
        assert all(r.rating == "good" for r in knee_results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
