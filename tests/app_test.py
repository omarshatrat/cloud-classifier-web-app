import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

parent_dir = Path(__file__).resolve().parent.parent
src_dir = parent_dir / 'src'
sys.path.append(str(parent_dir))

from app import load_model, predict

class TestCloudClassifier(unittest.TestCase):

    @patch('app.joblib.load')
    def test_load_model(self, mock_load):
        mock_load.return_value = MagicMock()
        model_name = 'rf'
        result = load_model(model_name)
        self.assertIsNotNone(result)

    def test_predict(self):
        mock_model = MagicMock()
        input_features = [[1, 2, 3, 4, 5]]
        result = predict(mock_model, input_features)
        self.assertIsNotNone(result)

    @patch('app.joblib.load')
    def test_load_model_invalid_input(self, mock_load):
        mock_load.side_effect = FileNotFoundError
        model_name = 'invalid_model_name'
        result = load_model(model_name)
        self.assertIsNone(result)

    def test_predict_invalid_input(self):
        mock_model = MagicMock()
        input_features = "invalid_input"
        result = predict(mock_model, input_features)
        self.assertIsNotNone(result) 


if __name__ == '__main__':
    unittest.main()
