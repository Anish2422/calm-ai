from django import forms
from ai.models import *

class AudioForm(forms.ModelForm):
    class Meta:
        model=Audio_store
        fields=['record']

class VideoForm(forms.Form):
    class Meta1:
        model=Video_store
        fields=['record']