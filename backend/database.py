from pymongo import MongoClient
from datetime import datetime

# MongoDB connection string (local MongoDB)
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "wildlife_detection"

# Collection names
DETECTIONS_COLLECTION = "detections"

# Connect to MongoDB
try:
    client = MongoClient(MONGO_URI)
    db = client[DATABASE_NAME]
    detections_collection = db[DETECTIONS_COLLECTION]
    print("✅ Connected to MongoDB successfully")
except Exception as e:
    print(f"❌ MongoDB connection error: {str(e)}")
    client = None
    db = None
    detections_collection = None


def store_detection(animal_name, detection_type, confidence, is_dangerous):
    """
    Store a detection record in MongoDB.

    Args:
        animal_name: Name of the detected animal
        detection_type: 'video' or 'audio'
        confidence: Detection confidence score
        is_dangerous: Boolean indicating if animal is dangerous

    Returns:
        Inserted document ID or None on error
    """
    if detections_collection is None:
        print("❌ MongoDB not connected, cannot store detection")
        return None

    try:
        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError):
            normalized_confidence = 0.0

        if normalized_confidence < 0.0:
            normalized_confidence = 0.0
        if normalized_confidence > 1.0:
            normalized_confidence = 1.0

        detection_record = {
            "animal_name": animal_name,
            "detection_type": detection_type,
            "confidence": normalized_confidence,
            "is_dangerous": bool(is_dangerous),
            "timestamp": datetime.now(),
        }

        result = detections_collection.insert_one(detection_record)
        print(f"✅ Stored detection: {animal_name} ({detection_type})")
        return result.inserted_id

    except Exception as e:
        print(f"❌ Error storing detection: {str(e)}")
        return None


def get_detection_history(limit=5):
    """
    Get the most recent detection history.

    Args:
        limit: Number of recent records to retrieve (default 5)

    Returns:
        List of detection records or empty list on error
    """
    if detections_collection is None:
        print("❌ MongoDB not connected, cannot fetch history")
        return []

    try:
        # Get most recent detections, sorted by timestamp descending
        cursor = detections_collection.find().sort("timestamp", -1).limit(limit)
        history = []

        for doc in cursor:
            raw_confidence = doc.get("confidence", 0.0)
            try:
                safe_confidence = float(raw_confidence)
            except (TypeError, ValueError):
                safe_confidence = 0.0
            if safe_confidence < 0.0:
                safe_confidence = 0.0
            if safe_confidence > 1.0:
                safe_confidence = 1.0

            history.append(
                {
                    "_id": str(doc["_id"]),
                    "animal_name": doc["animal_name"],
                    "detection_type": doc["detection_type"],
                    "confidence": safe_confidence,
                    "is_dangerous": doc["is_dangerous"],
                    "timestamp": doc["timestamp"].isoformat() if doc.get("timestamp") else None,
                }
            )

        return history

    except Exception as e:
        print(f"❌ Error fetching detection history: {str(e)}")
        return []


def get_all_detections():
    """
    Get all detection records from MongoDB.

    Returns:
        List of all detection records or empty list on error
    """
    if detections_collection is None:
        print("❌ MongoDB not connected")
        return []

    try:
        cursor = detections_collection.find().sort("timestamp", -1)
        history = []

        for doc in cursor:
            history.append(
                {
                    "_id": str(doc["_id"]),
                    "animal_name": doc["animal_name"],
                    "detection_type": doc["detection_type"],
                    "confidence": doc["confidence"],
                    "is_dangerous": doc["is_dangerous"],
                    "timestamp": doc["timestamp"].isoformat() if doc.get("timestamp") else None,
                }
            )

        return history

    except Exception as e:
        print(f"❌ Error fetching all detections: {str(e)}")
        return []
