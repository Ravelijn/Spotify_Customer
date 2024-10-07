import streamlit as st 
import streamlit.components.v1 as components 

# load the saved model 
pickle_in = open("fordeploy.pkl","rb")
model=pickle.load(pickle_in)


def predict_popu(acousticness,danceability,duration_ms,energy,loudness,speechiness,valence):
    """
    this method is for prediction process 
    takes all the Audio characteristics thtat we used for modelling and returns the prediction 
    """
    prediction=model.predict([[acousticness,danceability,duration_ms,energy,loudness,speechiness,valence]])
    print(prediction)
    return prediction



def main():
    st.title("Spotify songs")
     
    html_temp2 = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h2 style="color:white;text-align:center;">Spotify songsr </h2>
        <h1 style="color:white;text-align:center;">Popularity prediction</h1>
		</div>
		"""
    # a simple html code for heading which is in blue color and we can even use "st.write()" also ut for back ground color i used this HTML ..... 
    #  to render this we use ...
    components.html(html_temp2)
    # components.html() will render the render the 

    components.html("""
                <img src="https://www.tech-recipes.com/wp-content/uploads/2016/02/Spotify.png" width="700" height="150">
                
                """)
    # this is to insert the image the in the wed app simple <imag/> tag in HTML
    
    #now lets get the test input from the user by wed app 
    # for this we can use "st.text_input()" which allows use to get the input from the user 
    
    acousticness = st.text_input("acousticness","Type Here")
    danceability = st.text_input("danceability","Type Here")
    duration_ms = st.text_input("duration_ms","Type Here")
    energy = st.text_input("energy","Type Here")
    loudness = st.text_input("loudness","Type Here")
    speechiness = st.text_input("speechiness","Type Here")
    valence = st.text_input("valence","Type Here")
    result=""
    # done we got all the user inputs to predict and we need a button like a predict button we do that by "st.button()"
    # after hitting the button the prediction process will go on and then we print the success message by "st.success()"
    if st.button("Predict"):
        result=predict_popu(acousticness,danceability,duration_ms,energy,loudness,speechiness,valence)
    st.success('The Popularity of the song is {}'.format(result))
    # one more button saying About ...
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
