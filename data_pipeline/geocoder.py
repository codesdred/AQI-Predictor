# data_pipeline/geocoder.py
from geopy.geocoders import Nominatim
import streamlit as st

@st.cache_data(ttl=86400) # Cache coordinates for 24 hours to speed up app
def get_coordinates(city_name):
    """Converts a city name to latitude and longitude."""
    try:
        geolocator = Nominatim(user_agent="hyperlocal_aqi_app")
        location = geolocator.geocode(city_name)
        if location:
            return {"lat": location.latitude, "lon": location.longitude, "name": location.address}
        else:
            return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None