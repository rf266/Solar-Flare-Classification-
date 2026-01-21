from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI()

class Input(BaseModel): 
    activity: int
    evolution: int 
    previous_24_hour_flare_activity:int =Field(alias="previous 24 hour flare activity")
    historically_complex: int = Field(alias="historically-complex")
    became_complex_on_this_pass: int =Field(alias="became complex on this pass")
    area: int
    area_of_largest_spot: int =Field(alias="area of largest spot")
    modified_Zurich_class_B: bool =Field(alias="modified Zurich class_B")
    modified_Zurich_class_C: bool =Field(alias="modified Zurich class_C")
    modified_Zurich_class_D: bool =Field(alias="modified Zurich class_D")
    modified_Zurich_class_E: bool =Field(alias="modified Zurich class_E")
    modified_Zurich_class_F: bool =Field(alias="modified Zurich class_F")
    modified_Zurich_class_H: bool =Field(alias="modified Zurich class_H")
    largest_spot_size_A :bool = Field(alias="largest spot size_A")
    largest_spot_size_H:bool = Field(alias="largest spot size_H")
    largest_spot_size_K :bool = Field(alias="largest spot size_K")
    largest_spot_size_R :bool = Field(alias="largest spot size_R")
    largest_spot_size_S :bool = Field(alias="largest spot size_S")
    largest_spot_size_X :bool   = Field(alias="largest spot size_X")
    spot_distribution_C :bool = Field(alias="spot distribution_C")
    spot_distribution_I : bool = Field(alias="spot distribution_I")
    spot_distribution_O :bool = Field(alias="spot distribution_O")
    spot_distribution_X :bool = Field(alias="spot distribution_X")

@app.post("https://end-to-end-solar-flare-regression-model.onrender.com/")
def predict(data:Input):
    model = joblib.load("poisson_model.pkl")
    newdata = data.model_dump(by_alias=True)
    newdf = pd.DataFrame([newdata]) 
    prediction = pd.DataFrame(model.predict(newdf), columns=["common flares", "moderate flares", "severe flares"])

    return prediction.to_dict()
     