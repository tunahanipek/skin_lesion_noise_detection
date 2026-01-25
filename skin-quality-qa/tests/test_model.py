"""Tests for model functionality"""
import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import GOOGLE_API_KEY, IMAGE_ARTIFACTS, MODEL_NAME


class TestModelConfig(unittest.TestCase):
    """Test model configuration"""
    
    def test_api_key_exists(self):
        """API key should be configured"""
        self.assertIsNotNone(GOOGLE_API_KEY)
        self.assertGreater(len(GOOGLE_API_KEY), 0)
    
    def test_artifacts_defined(self):
        """Image artifacts should be defined"""
        self.assertEqual(len(IMAGE_ARTIFACTS), 6)
        self.assertIn("blurry", IMAGE_ARTIFACTS)
        self.assertIn("hairy", IMAGE_ARTIFACTS)
    
    def test_model_name_valid(self):
        """Model name should be set"""
        self.assertEqual(MODEL_NAME, "gemini-2.0-flash")


if __name__ == "__main__":
    unittest.main()
