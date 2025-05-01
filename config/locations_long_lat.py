"""
File contains information for the longitude and Latitude.


"""

def get_longitude_and_latitude_of_locations():
    """
    Gets the longitude and latitude from the config.

    """


    d = {
        ("Miami", "Florida"): (25.7752, -80.2086),
        ("Houston", "Texas"): (29.76, -95.37),
        ("Mobile", "Alabama"): (30.6684, -88.1002),
        ("Los Angeles", "California"): (34.0194, -118.4108),
        ("New York", "New York"): (40.6635, -73.9387),
        ("Chicago", "Illinois"): (41.8781, -87.623177),
        ("Kansas City", "Missouri"): (39.0997, -94.5786),
        ("Boston", "Massachusetts"): (42.3555, -71.0565),
        ("Washington", "DC"): (38.9072, -77.0369),
        ("Birmingham", "Alabama"): (33.51, -86.81),
        ("Seattle", "Washington"): (47.6061, -122.33),
        ("Philadelphia", "Pennsylvania"): (39.95, -75.16),
        ("Phoenix", "Arizona"): (33.4482, -112.0777) #A big addition to the model.
    }

    #d = {
    #    ("Houston", "Texas"): (29.76, -95.37),
    #    ("New York", "New York"): (40.6635, -73.9387),
    #    ("Dallas", "Texas"): (32.77, -96.79)
    #}

    #We can also look at the confusion matrix.


    return d
