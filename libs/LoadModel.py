# FILE NAME : LoadModel.py

from libs.ExecuteModel import execute_model

def TextToImage(Model, Prompt, Image):
    print ("Model Selected : ", Model)
    print ("Prompt Selected : ", Prompt)
    print ("Image Selected : ", Image)
    hasil = execute_model(Model, Prompt, Image)
    return hasil