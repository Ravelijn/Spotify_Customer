from fastapi import FastAPI
import pickle
import uvicorn
app = FastAPI()

#@app.get("/")
#def root():
#    return {"message": "Hello World"}

app = FastAPI()
pickle_in = open("fordeploy1.pkl","rb")
model = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Spotify Predictor API'}

@app.post('/Spotify/predict')
def predict_popu(acousticness,danceability,duration_ms,energy,loudness,speechiness,valence):
    """
    this method is for prediction process 
    takes all the Audio characteristics thtat we used for modelling and returns the prediction 
    """
    #try:
    #    prediction = 0
    #pickle_in = open("fordeploy1.pkl","rb")
    #model = pickle.load(pickle_in)
    prediction=model.predict([[acousticness,danceability,duration_ms,energy,loudness,speechiness,valence]])
    print(prediction)
    #except InconsistentVersionWarning as w:
    #    print(w.original_sklearn_version)
    return prediction

#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)