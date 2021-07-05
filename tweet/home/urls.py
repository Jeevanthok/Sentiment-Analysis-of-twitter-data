from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('live', views.livetweets, name='live'),
    path('about', views.about, name='about'),
    path('team', views.team, name='team'),
]
