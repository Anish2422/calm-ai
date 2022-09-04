import imp
import re
from django.shortcuts import render
from django.http import HttpResponse
from django.http.response import StreamingHttpResponse
from django.core.files.storage import default_storage
from ai.camera import VideoCamera
from ai.constants import *
from ai.forms import *
from ai.predict import *
from templates.ai import *
# Create your views here.

def index(response):
    return render(response, 'ai\index.html')

def meditate(response):
    return render(response, 'ai\meditate.html')

def gen(camera):
    while True:
        frame, finish = camera.get_frame(rpred, lpred, lbl, lbl_pred)
        if finish==False:
            break
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                    content_type='multipart/x-mixed-replace; boundary=frame')


def predict_emotions(request):
    if request.method == "POST":
        file = request.FILES["audioFile"]
        file_name = default_storage.save(file.name, file)
        file_url = str(default_storage.path(file_name))
        file_url= str('\\'.join(file_url.split('\\')[:-1])+'\\')
        print(file_url)
        print(file_name)
        ans = []
        if os.listdir(BASE_DIR+'\media\\')!=[]:
            print('Sound exists!!!!!')
            ans = app(k,str(file_url),str(file_name))
            print(ans)
        
        # ans = {file_name : ans}

        return render(request, "ai\predict.html", {"predictions": ans})

    else:
        return render(request, "ai\predict.html", {"predictions": 0})
    
    return render(request, "ai\predict.html", {"predictions": 0})