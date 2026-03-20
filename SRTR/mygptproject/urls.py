"""
URL configuration for mygptproject project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from gptapp import views

urlpatterns = [
    path('gpt-interface/', views.render_gpt_interface, name='gpt_interface'),
    #path('execute_python_code/', views.execute_python_code, name='execute_python_code'),
    path("api/query", views.api_query, name="api_query"),
    path('api/rebuild_index', views.api_rebuild_index),
    path('api/debug', views.api_debug),
    #path("api/execute", views.api_execute, name="api_execute"),
]

