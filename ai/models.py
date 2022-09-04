from __future__ import unicode_literals

from django.db import models

class Audio_store(models.Model):
    record=models.FileField(upload_to='documents/')
    class Meta:
        db_table='Audio_store'

class Video_store(models.Model):
    duration = models.IntegerField(help_text = "Enter the duration of your meditation (in seconds)")        
    class Meta1:
        db_table='Video_store'
