import unittest
import numpy as np
import tempfile
import os
import xml.etree.ElementTree as ET

# Import the functions we want to test
from feature_space_analysis import acf, dfa, rms, compute_sleep_distance, compute_state_velocity
from bifurcation_fitting_example import harvest
from data_loader import parse_nsrr_xml

class TestSleepAnalysis(unittest.TestCase):

    def test_rms(self):
        """Test Root Mean Square calculation."""
        # RMS of [3, 4] is sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        x = np.array([3.0, 4.0])
        self.assertAlmostEqual(rms(x), np.sqrt(12.5))
        
        # Test with NaNs
        x_nan = np.array([3.0, 4.0, np.nan])
        self.assertAlmostEqual(rms(x_nan), np.sqrt(12.5))

    def test_acf_lag1(self):
        """Test Lag-1 Autocorrelation (acf)."""
        # Testing with a simple linear series
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Expected value can be calculated or compared against a known baseline
        res = acf(x, 1)
        self.assertTrue(-1.0 <= res <= 1.0)
        
        # Constant series has variance 0, should return NaN
        self.assertTrue(np.isnan(acf(np.array([1.0, 1.0, 1.0]))))

    def test_dfa(self):
        """Test Detrended Fluctuation Analysis (dfa)."""
        # A simple test to make sure it runs and returns a number for sufficiently long inputs
        x = np.sin(np.linspace(0, 10, 100))
        res = dfa(x, smin=10, sstep=2)
        self.assertTrue(np.isreal(res))
        
        # Short input should return NaN
        short_x = np.array([1.0, 2.0])
        self.assertTrue(np.isnan(dfa(short_x)))

    def test_harvest_ode(self):
        """Test the bifurcation model ODE definition."""
        # dx/dt = r*x*(1 - x/K) - c*(x^2 / (x^2 + h^2))
        # dc/dt = m
        t = 0
        y = [2.0, 1.0] # x, c
        r, K, h, m = 3.0, 10.0, 0.5, 0.1
        
        dxdt, dcdt = harvest(t, y, r, K, h, m)
        
        expected_dxdt = 3.0 * 2.0 * (1 - 2.0/10.0) - 1.0 * (4.0 / (4.0 + 0.25))
        expected_dcdt = 0.1
        
        self.assertAlmostEqual(dxdt, expected_dxdt)
        self.assertAlmostEqual(dcdt, expected_dcdt)

    def test_compute_sleep_distance(self):
        """Test Sleep Distance calculation."""
        # 3 epochs, 2 features
        features = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ])
        # Let's say the last epoch is the sleep onset period
        onset_indices = [2]
        
        # Centroid is [3.0, 3.0]
        # Distances:
        # Ep 0: sqrt((1-3)^2 + (1-3)^2) = sqrt(8) ≈ 2.828
        # Ep 1: sqrt((2-3)^2 + (2-3)^2) = sqrt(2) ≈ 1.414
        # Ep 2: 0.0
        distances = compute_sleep_distance(features, onset_indices)
        
        self.assertAlmostEqual(distances[0], np.sqrt(8))
        self.assertAlmostEqual(distances[1], np.sqrt(2))
        self.assertAlmostEqual(distances[2], 0.0)

    def test_compute_state_velocity(self):
        """Test State Velocity calculation."""
        # 3 epochs, 2 features
        features = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [4.0, 4.0]
        ])
        
        # Velocity at ep 0: NaN (no previous)
        # Velocity at ep 1: dist(ep0, ep1) = sqrt(2) ≈ 1.414
        # Velocity at ep 2: dist(ep1, ep2) = sqrt(8) ≈ 2.828
        velocity = compute_state_velocity(features)
        
        self.assertTrue(np.isnan(velocity[0]))
        self.assertAlmostEqual(velocity[1], np.sqrt(2))
        self.assertAlmostEqual(velocity[2], np.sqrt(8))

    def test_parse_nsrr_xml(self):
        """Test NSRR XML parsing using a mock XML structure."""
        # Create a mock XML file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
        <PSGAnnotation>
            <ScoredEvents>
                <ScoredEvent>
                    <EventConcept>Stage 1 sleep|1</EventConcept>
                    <Start>0</Start>
                    <Duration>30</Duration>
                </ScoredEvent>
                <ScoredEvent>
                    <EventConcept>Stage 2 sleep|2</EventConcept>
                    <Start>30</Start>
                    <Duration>60</Duration>
                </ScoredEvent>
                <ScoredEvent>
                    <EventConcept>Arousal</EventConcept>
                    <Start>45</Start>
                    <Duration>5</Duration>
                </ScoredEvent>
            </ScoredEvents>
        </PSGAnnotation>
        """
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xml', mode='w', encoding='utf-8') as temp_xml:
            temp_xml.write(xml_content)
            temp_path = temp_xml.name
            
        try:
            result = parse_nsrr_xml(temp_path, epoch_len=30)
            self.assertIsNotNone(result)
            # Expecting 3 epochs (30s each: 1 at stage 1, 2 at stage 2)
            # Epoch 0 (0-30s): Stage 1
            # Epoch 1 (30-60s): Stage 2
            # Epoch 2 (60-90s): Stage 2
            np.testing.assert_array_equal(result['scoringLabels'], [1, 2, 2])
            np.testing.assert_array_equal(result['ArousalsStart'], [45.0])
            np.testing.assert_array_equal(result['ArousalsDuration'], [5.0])
        finally:
            os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
