from django.contrib import admin
from django.urls import path
from detector.views import video_feed, home  # Import the 'home' view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),  # Home Page
    path('video-feed/', video_feed, name='video_feed'),  # Live Webcam Page
]
