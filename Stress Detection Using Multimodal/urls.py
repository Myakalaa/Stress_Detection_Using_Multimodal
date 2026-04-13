"""
URL configuration for Epileptic Seizure Detection EEG project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
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
from admins import views as admins
from django.urls import path
from users import views as usr
from . import views as mainView
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    # Django Admin Panel
    path('admin/', admin.site.urls),

    # Main / Common Pages
    path('', mainView.index, name='index'),
    path('index/', mainView.index, name='index'),
    path('logout/', mainView.logout, name='logout'),
    path('please-login/', mainView.please_login, name='please_login'),

    # Admin Section
    path('AdminLogin/', admins.AdminLogin, name='AdminLogin'),
    path('AdminLoginCheck/', admins.AdminLoginCheck, name='AdminLoginCheck'),
    path('AdminHome/', admins.AdminHome, name='AdminHome'),
    path('ViewRegisteredUsers/', admins.ViewRegisteredUsers, name='ViewRegisteredUsers'),
    path('AdminActivaUsers/', admins.AdminActivaUsers, name='AdminActivaUsers'),
    path('DeleteUsers/', admins.DeleteUsers, name='DeleteUsers'),

    # User Section
    path('UserLogin/', usr.UserLogin, name='UserLogin'),
    path('UserLoginCheck/', usr.UserLoginCheck, name='UserLoginCheck'),
    path('UserRegisterForm/', usr.UserRegisterForm, name='UserRegisterForm'),
    path('UserRegisterActions/', usr.UserRegisterActions, name='UserRegisterActions'),
    path('Userbase/', usr.Userbase, name='Userbase'),
    path('training/', usr.Training, name='Training'),
    path("prediction/", usr.prediction_home, name="prediction_home"),
    path("prediction/csv/", usr.stress_prediction_csv, name="stress_prediction_csv"),
    path("prediction/manual/", usr.stress_prediction_manual, name="stress_prediction_manual"),
]



urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.UPLOADS_URL, document_root=settings.UPLOADS_ROOT)

