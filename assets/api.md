# API Documentation

## Analyze Traffic

POST /api/analyze

### Request

```json
{
  "camera_id": "CAM001"
}
```

### Response

```json
{
  "traffic_density": "High",
  "vehicle_count": 245,
  "congestion_score": 87
}
```

---

## Predict Congestion

POST /api/predict

### Response

```json
{
  "prediction": "Heavy Traffic",
  "confidence": "92%"
}
```
