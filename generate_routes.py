import json
import requests
import sys
import os

# Use Google Maps Routing API to calculate approximate routes between citibike stations
# Usage: API_KEY=... python3 generate_routes.py citibike_history.json

API_KEY = os.getenv("API_KEY")
assert API_KEY is not None, "Set the API_KEY environment variable with your Google Maps API key"

citibike_stations = {}
route_cache = {}


def populate_citibike_stations():
    url = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    for station in data["data"]["stations"]:
        citibike_stations[station["name"]] = station


def get_bike_route_geojson(
    origin,
    destination,
    api_key,
):
    if (origin, destination) in route_cache:
        return route_cache[(origin, destination)]
    if (destination, origin) in route_cache:
        return route_cache[(destination, origin)]
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "routes.distanceMeters,routes.polyline",
    }

    body = {
        "origin": {"location": {"latLng": {"latitude": origin[0], "longitude": origin[1]}}},
        "destination": {
            "location": {"latLng": {"latitude": destination[0], "longitude": destination[1]}}
        },
        "travelMode": "BICYCLE",
        "polylineEncoding": "GEO_JSON_LINESTRING",
    }

    r = requests.post(url, json=body, headers=headers)
    r.raise_for_status()
    data = r.json()

    route = data["routes"][0]
    poly = route["polyline"]["geoJsonLinestring"]
    result = {
        "type": "Feature",
        "properties": {
            "distance_m": route["distanceMeters"],
        },
        "geometry": poly,
    }
    route_cache[(origin, destination)] = result
    return result


if __name__ == "__main__":
    citibike_file = sys.argv[1]
    citibike_rides = []
    with open(citibike_file, "r") as f:
        citibike_rides = json.load(f)
    populate_citibike_stations()
    routes = []
    for ride in citibike_rides:
        # start and end station are the same, we can't calculate anything
        if ride["startAddress"] == ride["endAddress"]:
            continue
        origin = citibike_stations.get(ride["startAddress"])
        destination = citibike_stations.get(ride["endAddress"])
        # the station doesn't exist
        if not origin or not destination:
            continue
        route = get_bike_route_geojson(
            (origin["lat"], origin["lon"]),
            (destination["lat"], destination["lon"]),
            API_KEY
        )
        routes.append(route)
    result = {
        "type": "FeatureCollection",
        "features": routes,
    }
    with open("citibike_routes.json", "w") as f:
        json.dump(result, f)
