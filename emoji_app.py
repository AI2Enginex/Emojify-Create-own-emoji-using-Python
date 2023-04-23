
'''
Model Deployment using Tkinter Python

'''
import cv2                               # using opencv library to read inputs from webcam
import numpy as np                                       
import tkinter as tk                     # creating GUI widgets using tkinter library in python
from keras.models import load_model      # for loading the CNN model


# creating a user defined class Parameters which contains all the variables
# required for testing the model

class Parameters:

    def __init__(self):

        # loading the CNN  Model
        self.model = load_model('./emoji_face.h5')

        # creating list of labels to predict
        self.labels = ["Angry", "Disgusted", "Fearful",
                       "Happy", "Neutral", "Sad", "Surprised"]

        # creating a list of emojies to map with emotions
        self.emoji_dist = ["./emojis/angry.png",  "./emojis/disgused.png", "./emojis/fear.png", "./emojis/happy.png",
                           "./emojis/neutral.png", "./emojis/Sad.png", "./emojis/surprise.png"]

        # haarcascade for detecting face
        self.cascade_classifier = cv2.CascadeClassifier(
            './haarcascade_frontalface_default.xml')


# creating a user defined class Map_Emojies to evaluate
# the model's performance on live camera
# inherit the parent class (Parameters) to load all the
# required variables

# derived class Map_Emojies inherits base class Parameters
class Map_Emojies(Parameters):

    def __init__(self):

        super().__init__()               # inheriting base class constructor
        self.show_text = 0               # declaring a integer variable for keeping
        self.cap = cv2.VideoCapture(0)   # using VideoCapture to read webcam
        # the parameter (0) in VideoCapture represents webcam

    # creating a function predict_emotions to capture
    # and predict emotions on live webcam images

    def predeict_emotions(self):

        while True:

            ret, frame = self.cap.read()
            faces = self.cascade_classifier.detectMultiScale(frame)
            for (x, y, w, h) in faces:

                # creating a bounding box on the predicted face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                # creating region of intrest
                roi_ = frame[y:y+h, x:x+w]
                roi_ = cv2.resize(roi_, (48, 48),                            # as input for the model
                                  interpolation=cv2.INTER_AREA)

                if np.sum([roi_]) != 0:
                    # scaling images between 0 to 1
                    roi = roi_.astype('float')/255.0
                    roi = np.expand_dims(roi, axis=0)

                    # predicting emotion on the roi
                    prediction = self.model.predict(roi)[0]

                    # mapping prediction to it's corresponding label
                    label = self.labels[prediction.argmax()]
                    self.show_text = prediction.argmax()
                    # using generator to keep tarck of emotions
                    yield self.show_text
                    # predicted while capturing webcam
                    label_position = (x, y)

                    cv2.putText(frame, label, label_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Emotion Detector', cv2.resize(frame,(350,300)))
            cv2.moveWindow('Emotion Detector', 400, 300)

            # break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()

        cv2.destroyAllWindows()

    # creating a function display_emoji to display
    # the emojies for the predicted emotion

    def display_emoji(self):

        for data in self.predeict_emotions():                          # call the predict_emotion function

            # read the image for the predicted emotion label
            image = cv2.resize(cv2.imread(self.emoji_dist[data]), (300, 300))

            cv2.imshow('Photo to emoji', image)
            cv2.moveWindow('Photo to emoji', 800, 300)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


class GUI_Application:

    def __init__(self):

        # creating a tkinter window to run the application
        self.root = tk.Tk()
        self.root.title('Emojify - Create your own Emoji')
        self.root['bg'] = 'black'
        self.root.geometry('900x600+300+50')
        tk.Label(self.root, text='Photo to Emoji',
                 font=('Times New Roman', 40),fg='white',bg='black',pady=20).pack()

    def run_app(self):

        # creating class object for class Map_Emojies
        # calling the function display_emoji for
        # Evaluating model performance for webcam capture

        me = Map_Emojies()
        me.display_emoji()

    def buttons(self):

        tk.Button(self.root, text='Open Camera', command=self.run_app).pack()
        self.root.mainloop()


if __name__ == '__main__':

    app = GUI_Application()
    app.buttons()
