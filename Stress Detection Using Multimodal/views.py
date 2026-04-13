from django.shortcuts import render, redirect
from users.forms import UserRegistrationForm
from django.contrib import messages

# COMMON VIEWS

# Create your views here.
def index(request):
    return render(request, 'index.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})

def please_login(request):
    return render(request, 'please_login.html')

def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def logout(request):
    # ✅ This clears all session data completely
    request.session.flush()
    
    # ✅ Optional: also delete session cookie
    request.session.clear_expired()
    
    messages.success(request, "You have been logged out successfully!")
    return redirect('UserLogin')







