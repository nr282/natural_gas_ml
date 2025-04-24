import unittest
from weather import get_all_temperature_data
import datetime

class TestStringMethods(unittest.TestCase):

    def test_weather_data(self):
        start = datetime.datetime(2018, 1, 1)
        end = datetime.datetime(2018, 12, 31)
        df = get_all_temperature_data(start, end)
        self.assertTrue(len(df) > 0)
        self.assertTrue(len(df.City.unique()) > 0)
        self.assertTrue(len(df.State.unique()) > 0)


if __name__ == '__main__':
    unittest.main()