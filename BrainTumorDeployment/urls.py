from django.contrib import admin
from django.urls import path, include,re_path

from django.conf.urls.static import static
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path('', include('BrainTumorApp.urls')),
    path('admin/', admin.site.urls),
    re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
]

# if settings.DEBUG:
#     # setting this to view media files from admin panel
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
#     urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)