"""
In the following code base, I examine weather related APIs.

The weather APIs that were investigated have been:
    1. python-weather
    2. meteostat-python

"""

import python_weather
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import pandas as pd
from config.locations_long_lat import get_longitude_and_latitude_of_locations
import os


def example_usage_of_meteostat():
    """
    This is an example of collecting weather data using the meteostat API.


    """

    # Import Meteostat library and dependencies
    from datetime import datetime
    import matplotlib.pyplot as plt
    from meteostat import Point, Daily

    # Set time period
    start = datetime(2018, 1, 1)
    end = datetime(2018, 12, 31)

    # Create Point for Vancouver, BC
    location = Point(49.2497, -123.1193, 70)

    # Get daily data for 2018
    data = Daily(location, start, end)
    data = data.fetch()

    # Plot line chart including average, minimum and maximum temperature
    data.plot(y=['tavg', 'tmin', 'tmax'])
    plt.show()




def get_all_temperature_data(start: datetime,
                            end: datetime) -> pd.DataFrame:

    dataframes = []
    longitude_and_latitude_of_locations = get_longitude_and_latitude_of_locations()
    for location in longitude_and_latitude_of_locations:
        city, state = location
        long, lat = longitude_and_latitude_of_locations.get(location)
        loc_point = Point(long, lat)
        data = Daily(loc_point, start, end)
        data = data.fetch()

        data["City"] = city
        data["State"] = state

        dataframes.append(data)

    complete_data = pd.concat(dataframes)
    complete_data.to_csv(os.path.join("data", "weather", "weather_data.csv"))
    return complete_data












if __name__ == '__main__':

    example_usage_of_meteostat()






