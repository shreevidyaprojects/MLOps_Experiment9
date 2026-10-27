import unittest
import pandas as pd

from model_pipeline import load_data, train_model, predict_marks


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = load_data("student_performance.csv")

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.data), 0)

    def test_required_columns(self):
        required = [
            "StudyHours", "Attendance",
            "PreviousMarks", "FinalMarks"
        ]

        for column in required:
            self.assertIn(column, self.data.columns)

    def test_invalid_columns_rejected(self):
        invalid_data = pd.DataFrame({
            "StudyHours": [3],
            "Attendance": [70]
        })

        with self.assertRaises(ValueError):
            train_model(invalid_data)

    def test_empty_data_rejected(self):
        with self.assertRaises(ValueError):
            train_model(self.data.iloc[0:0])

    def test_missing_values_rejected(self):
        invalid_data = self.data.copy()
        invalid_data.loc[0, "StudyHours"] = None

        with self.assertRaises(ValueError):
            train_model(invalid_data)

    def test_model_trains(self):
        model = train_model(self.data)
        self.assertIsNotNone(model)

    def test_prediction_returns_one_value(self):
        model = train_model(self.data)

        result = predict_marks(model, 6, 85, 65)

        self.assertEqual(len(result), 1)
        self.assertTrue(pd.notna(result[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
