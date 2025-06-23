import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from pipelines.train_pipeline import transform_data

class TestPipeline(unittest.TestCase):
    def test_transform_data(self):
        data = pd.DataFrame({
            "driver_id": [101],
            "deliveries_completed": [150],
            "hours_worked": [None],
            "earnings": [5000],
            "city": ["Bengaluru"],
            "tenure_months": [6],
            "churn": [0]
        })
        transformed = transform_data(data)
        self.assertEqual(transformed["city_Bengaluru"].iloc[0], 1)
        self.assertFalse(transformed["hours_worked"].isnull().any())
        self.assertIn("earnings_per_delivery", transformed.columns)

if __name__ == "__main__":
    unittest.main()