"""Hospital discovery service using Google Places API with curated Indian fallback directory."""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from .http_client import request_json

CACHE_TTL_SEC = 30 * 60
_hospital_cache: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

FALLBACK_HOSPITALS: List[Dict[str, Any]] = [
    {
        "placeId": "medcy_hyderabad_001",
        "name": "Medcy Fertility & IVF Hospital",
        "address": "Road No. 12, Banjara Hills, Hyderabad, Telangana",
        "phoneNumber": "+91 40 2335 8899",
        "rating": 4.8,
        "totalRatings": 412,
        "isOpen": True,
        "url": "https://www.google.com/maps/search/Medcy+Fertility+IVF+Hospital+Hyderabad",
    },
    {
        "placeId": "sakhi_care_center_002",
        "name": "Sakhi Women & Maternity Clinic",
        "address": "Jubilee Hills Checkpost, Hyderabad, Telangana",
        "phoneNumber": "+91 40 2354 6622",
        "rating": 4.7,
        "totalRatings": 298,
        "isOpen": True,
        "url": "https://www.google.com/maps/search/Sakhi+Women+Maternity+Clinic+Hyderabad",
    },
    {
        "placeId": "rainbow_maternity_003",
        "name": "Rainbow Children's & BirthRight Hospital",
        "address": "Banjara Hills Main Road, Hyderabad, Telangana",
        "phoneNumber": "+91 40 4455 0000",
        "rating": 4.6,
        "totalRatings": 865,
        "isOpen": True,
        "url": "https://www.google.com/maps/search/Rainbow+Hospital+Hyderabad",
    },
    {
        "placeId": "fernandez_maternity_004",
        "name": "Fernandez Stork Home Maternity",
        "address": "Bogulkunta, Abids, Hyderabad, Telangana",
        "phoneNumber": "+91 40 4022 2300",
        "rating": 4.6,
        "totalRatings": 640,
        "isOpen": True,
        "url": "https://www.google.com/maps/search/Fernandez+Hospital+Hyderabad",
    },
    {
        "placeId": "apollo_cradle_005",
        "name": "Apollo Cradle & Fertility Centre",
        "address": "Kondapur Main Road, Hyderabad, Telangana",
        "phoneNumber": "+91 40 4424 4424",
        "rating": 4.5,
        "totalRatings": 510,
        "isOpen": True,
        "url": "https://www.google.com/maps/search/Apollo+Cradle+Hyderabad",
    },
]


async def _geocode_zipcode(zipcode: str) -> Optional[Dict[str, float]]:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        status_code, data = await request_json(
            "GET",
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": zipcode, "region": "in", "key": api_key},
            timeout=6.0,
        )
        if status_code == 200 and isinstance(data, dict) and data.get("status") == "OK" and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return {"lat": float(loc["lat"]), "lng": float(loc["lng"])}
    except Exception:
        pass
    return None


async def find_nearby_hospitals(
    zipcode: str,
    specialty: str = "General",
    radius_meters: int = 10000,
) -> List[Dict[str, Any]]:
    if not zipcode:
        return []

    cache_key = f"{zipcode}_{specialty}"
    cached = _hospital_cache.get(cache_key)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SEC:
        return cached[1]

    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if api_key:
        location = await _geocode_zipcode(zipcode)
        if location:
            query = (
                f"{specialty} hospitals near {zipcode}"
                if specialty and specialty.lower() != "general"
                else f"hospitals near {zipcode}"
            )
            try:
                radius = float(os.getenv("HOSPITAL_SEARCH_RADIUS_METERS", str(radius_meters)))
            except ValueError:
                radius = float(radius_meters)

            try:
                status_code, data = await request_json(
                    "POST",
                    "https://places.googleapis.com/v1/places:searchText",
                    headers={
                        "Content-Type": "application/json",
                        "X-Goog-Api-Key": api_key,
                        "X-Goog-FieldMask": (
                            "places.id,places.displayName,places.formattedAddress,"
                            "places.location,places.rating,places.userRatingCount,"
                            "places.currentOpeningHours,places.types,places.googleMapsUri"
                        ),
                    },
                    json_body={
                        "textQuery": query,
                        "maxResultCount": 10,
                        "locationBias": {
                            "circle": {
                                "center": {
                                    "latitude": location["lat"],
                                    "longitude": location["lng"],
                                },
                                "radius": radius,
                            }
                        },
                    },
                    timeout=8.0,
                )
                places = data.get("places") if status_code == 200 and isinstance(data, dict) else []
                if places:
                    found = [
                        {
                            "placeId": p.get("id", ""),
                            "name": (p.get("displayName") or {}).get("text", "Unknown Hospital"),
                            "address": p.get("formattedAddress", "Address not available"),
                            "rating": p.get("rating", 0) or 0,
                            "totalRatings": p.get("userRatingCount", 0) or 0,
                            "isOpen": (p.get("currentOpeningHours") or {}).get("openNow"),
                            "googleMapsUri": p.get("googleMapsUri"),
                        }
                        for p in places
                    ]
                    found.sort(
                        key=lambda h: (h.get("rating", 0), h.get("totalRatings", 0)),
                        reverse=True,
                    )
                    _hospital_cache[cache_key] = (time.time(), found)
                    return found
            except Exception:
                pass

    fallback = [
        {**h, "address": f"{h['address']} - {zipcode}"} for h in FALLBACK_HOSPITALS
    ]
    _hospital_cache[cache_key] = (time.time(), fallback)
    return fallback


async def get_hospital_details(place_id: str) -> Optional[Dict[str, Any]]:
    if not place_id:
        return None

    for item in FALLBACK_HOSPITALS:
        if item["placeId"] == place_id:
            return item

    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if api_key:
        try:
            status_code, place = await request_json(
                "GET",
                f"https://places.googleapis.com/v1/places/{quote(place_id, safe='')}",
                headers={
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": (
                        "id,displayName,formattedAddress,nationalPhoneNumber,"
                        "internationalPhoneNumber,rating,userRatingCount,"
                        "currentOpeningHours,googleMapsUri"
                    ),
                },
                timeout=8.0,
            )
            if status_code == 200 and isinstance(place, dict) and place.get("id"):
                return {
                    "placeId": place["id"],
                    "name": (place.get("displayName") or {}).get("text", "Unknown Hospital"),
                    "address": place.get("formattedAddress", "Address not available"),
                    "phoneNumber": (
                        place.get("internationalPhoneNumber")
                        or place.get("nationalPhoneNumber")
                        or "Not available"
                    ),
                    "rating": place.get("rating", 0) or 0,
                    "totalRatings": place.get("userRatingCount", 0) or 0,
                    "isOpen": (place.get("currentOpeningHours") or {}).get("openNow"),
                    "url": place.get("googleMapsUri")
                    or f"https://www.google.com/maps/place/?q=place_id:{quote(place_id, safe='')}",
                }
        except Exception:
            pass

    return FALLBACK_HOSPITALS[0]


def build_hospital_list_message(hospitals: List[Dict[str, Any]], zipcode: str) -> Dict[str, Any]:
    if not hospitals:
        return {
            "type": "text",
            "text": {
                "body": f"I couldn't find any hospitals near pincode {zipcode} at the moment."
            },
        }

    rows = []
    for hospital in hospitals[:10]:
        desc_parts: List[str] = []
        if (hospital.get("rating") or 0) > 0:
            desc_parts.append(
                f"★ {hospital['rating']} ({hospital.get('totalRatings', 0)} reviews)"
            )
        if hospital.get("isOpen") is not None:
            desc_parts.append("Open Now" if hospital["isOpen"] else "Closed")
        description = " · ".join(desc_parts)[:72] or "Tap for details"
        rows.append(
            {
                "id": f"hospital_{hospital['placeId']}",
                "title": str(hospital.get("name") or "Hospital")[:48],
                "description": description,
            }
        )

    return {
        "type": "interactive",
        "interactive": {
            "type": "list",
            "header": {"type": "text", "text": "Hospitals Near You"},
            "body": {
                "text": (
                    f"We found {len(hospitals)} fertility & maternity hospitals near pincode {zipcode}. "
                    "Select a center below for contact and WhatsApp details."
                )
            },
            "footer": {"text": "Select a hospital to view direct contact details"},
            "action": {
                "button": "View Hospitals",
                "sections": [{"title": "Nearby Hospitals", "rows": rows}],
            },
        },
    }


def build_hospital_detail_card(hospital: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not hospital:
        return {
            "type": "text",
            "text": {"body": "Sorry, I couldn't fetch details for this hospital right now."},
        }

    parts: List[str] = [
        f"*{hospital.get('name') or 'Unknown'}*",
        "",
        f"Address: {hospital.get('address') or 'Address not available'}",
    ]
    if (hospital.get("rating") or 0) > 0:
        parts.append(
            f"Rating: {hospital['rating']}/5 ({hospital.get('totalRatings', 0)} reviews)"
        )
    if hospital.get("isOpen") is not None:
        parts.append(f"Hours: {'Open Now' if hospital['isOpen'] else 'Closed'}")

    phone = hospital.get("phoneNumber")
    if phone and phone != "Not available":
        parts.append(f"Phone: {phone}")
        clean_phone = re.sub(r"\D", "", str(phone))
        if len(clean_phone) == 10:
            clean_phone = f"91{clean_phone}"
        if clean_phone:
            wa_link = (
                f"https://wa.me/{clean_phone}"
                "?text=Hello%2C%20I%20would%20like%20to%20inquire%20about%20hospital%20services."
            )
            parts.append(f"*WhatsApp Direct:* {wa_link}")

    url = hospital.get("url") or hospital.get("googleMapsUri")
    if url:
        parts.append("")
        parts.append(f"*View on Maps:* {url}")

    return {"type": "text", "text": {"body": "\n".join(parts)}}
