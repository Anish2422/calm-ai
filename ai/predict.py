import math
import os
import fleep
from pydub import AudioSegment
import pydub
import pickle
import pandas as pd
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import librosa
from med_app.settings import BASE_DIR
pydub.AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"
k1=0

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_t, to_t, split_filename):
        t1 = from_t * 1000
        t2 = to_t * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")
        
    def multiple_split(self,segment_length):
        total_t = math.ceil(self.get_duration())
        for i in range(0, total_t,segment_length):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+segment_length, split_fn)
            print(str(i) + ' Done')
            if i == total_t - segment_length:
                print('All splited successfully')
        return(i)

def get_audio(k,folder,file):
  # folder = str(input("Enter Folder Directory: "))

  # folder = "C:\Development\Mini Project\meditation_app\media\documents"

  if str(os.path.isdir(folder)) =='True' :
    print("True")
    f=0
    while (f==0):
      # file = str(input("Enter File Name: "))

      # file = 'sample.mp3'
      print(folder + file)
      if str(os.path.isfile(folder + file)) == 'True':
        with open(folder + file, "rb") as fi:
          print("True1")
          info = fleep.get(fi.read(128))
        if info.extension[0] == 'mp3' :
          src = folder + file
          dst = src + '.wav'
          # convert mp3 to wav
          sound = AudioSegment.from_mp3(src)
          sound.export(dst, format="wav")
          file=file+'.wav'
          split_wav = SplitWavAudioMubin(folder, file)
          k = split_wav.multiple_split(segment_length=5)
          f=1
        elif info.extension[0] == 'm4a' or info.extension[0] == 'wav':
          split_wav = SplitWavAudioMubin(folder, file)
          k = split_wav.multiple_split(segment_length=5)
          f=1
        else:
          print("Please enter name of audio file with supported a format (mp3,m4a,wav)") 
      else:
        print("Please ensure that the file is in the directory specified above and verify the file name")
        break
    return(k, file)
  else:
    print('Check if directory is correct') 
    get_audio()

def app(k,folder,file):

    k, file = get_audio(k1,folder,file)
    n = k/5
    ans = {}
    json_file = open(BASE_DIR + '\\ai\static\\ai\\files\model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(BASE_DIR + '\\ai\static\\ai\\files\Emotion_Model.h5')
    print("Model Loaded!")

    opt = tf.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    files = [file]

    for i in range(int(n)):
    # Transform Dataset
        X, sample_rate = librosa.load(folder + str(5*i) + '_' + file,res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)
        files.append(str(5*i) + '_' + file)
        print(str(5*i) + '_' + file)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        newdf = pd.DataFrame(data=mfccs).T

        # Predictions
        newdf= np.expand_dims(newdf, axis=2)
        newpred = loaded_model.predict(newdf, 
                                batch_size=16, 
                                verbose=1)

        filename = BASE_DIR + '\\ai\static\\ai\\files\labels'
        infile = open(filename,'rb')

        lb = pickle.load(infile)
        infile.close()

        # Final Prediction
        final = newpred.argmax(axis=1)
        final = final.astype(int).flatten()
        final = (lb.inverse_transform((final)))
        print(f"FOR {i} emotion is {final[0]}")
        ans[(i+1)*5] = final[0]
        print('Predicted label:',final) 
    
    try:
      files.append(str(5*(i+1)) + '_' + file)
      files.append(file.replace('.wav',''))
    except:
      files.append(file.replace('.wav',''))

    if n==0:
      if os.path.exists(folder+'0_'+file):
        os.remove(folder+'0_'+file)

    for i in files:
      if os.path.exists(folder+i):
        os.remove(folder+i)

    return ans

    