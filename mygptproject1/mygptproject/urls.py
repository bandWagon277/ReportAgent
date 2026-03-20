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
    path('process-csv/', views.process_csv, name='process_csv'),
    path('upload_csv/', views.upload_csv, name='upload_csv'),
    path('preview_result/', views.preview_result, name='preview_result'),
    path('download_result/', views.download_result, name='download_result'),
    path('upload_files/', views.upload_files, name='upload_files'),
    path('execute_pdf_pipeline/', views.execute_pdf_pipeline, name='execute_pdf_pipeline'),
    path('save_agent_a_code/', views.save_agent_a_code, name='save_agent_a_code'),
    path('upload_and_analyze_text/', views.upload_and_analyze_text),
    path('process_text/', views.process_text),
]

from gptapp import synthetic_data_views

# 在 urlpatterns 列表中添加
urlpatterns += [
    path('synthetic/', synthetic_data_views.synthetic_data_interface),
    path('api/analyze-synthetic/', synthetic_data_views.analyze_synthetic_data_request),
    path('api/generate-synthetic/', synthetic_data_views.generate_synthetic_data),
    path('api/download-synthetic/', synthetic_data_views.download_synthetic_data),
]