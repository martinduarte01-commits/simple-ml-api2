"""
FastAPI application for wine customer segmentation.
"""

from pathlib import Path

import joblib
import numpy as np

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------

MODEL_PATH = Path("models/model.joblib")


if MODEL_PATH.exists():

    model_bundle = joblib.load(
        MODEL_PATH
    )

    model = model_bundle["model"]

    offer_ids = model_bundle[
        "offer_ids"
    ]

else:

    model = None
    offer_ids = []


# ---------------------------------------------------------
# 2. Create FastAPI app
# ---------------------------------------------------------

app = FastAPI(
    title="Wine Customer Segmentation API",
    description=(
        "Predict which customer segment a customer "
        "belongs to based on the wine offers purchased."
    )
)


# ---------------------------------------------------------
# 3. Input format
# ---------------------------------------------------------

class PredictionInput(BaseModel):

    offer_ids: list[int] = Field(
        ...,
        min_length=1
    )

    class Config:

        json_schema_extra = {
            "example": {
                "offer_ids": [1, 4, 7]
            }
        }


# ---------------------------------------------------------
# 4. Output format
# ---------------------------------------------------------

class PredictionOutput(BaseModel):

    cluster: int
    distance_to_centroid: float


# ---------------------------------------------------------
# 5. Root endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "number_of_offer_features": len(
            offer_ids
        )
    }


# ---------------------------------------------------------
# 6. Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ---------------------------------------------------------
# 7. Prediction endpoint
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(data: PredictionInput):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    # ---------------------------------------------
    # Validate offer IDs
    # ---------------------------------------------

    unknown_offers = [
        offer_id
        for offer_id in data.offer_ids
        if offer_id not in offer_ids
    ]


    if unknown_offers:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown offer IDs: "
                f"{unknown_offers}"
            )
        )


    # ---------------------------------------------
    # Create customer feature vector
    # ---------------------------------------------

    customer_vector = np.zeros(
        len(offer_ids)
    )


    offer_position = {
        offer_id: i
        for i, offer_id
        in enumerate(offer_ids)
    }


    for offer_id in data.offer_ids:

        i = offer_position[
            offer_id
        ]

        customer_vector[i] += 1


    # KMeans expects a 2-D array
    customer_vector = (
        customer_vector
        .reshape(1, -1)
    )


    # ---------------------------------------------
    # Predict cluster
    # ---------------------------------------------

    cluster = model.predict(
        customer_vector
    )[0]


    # Calculate distance from customer
    # to the assigned cluster center
    distances = model.transform(
        customer_vector
    )[0]

    distance_to_centroid = distances[
        cluster
    ]


    return PredictionOutput(
        cluster=int(cluster),
        distance_to_centroid=float(
            distance_to_centroid
        )
    )
