from django.contrib import messages
from django.shortcuts import redirect, render
from users.models import UserRegistrationModel
from users.views import user_login_required
# ADMIN VIEWS

def AdminLogin(request):
    return render(request, 'AdminLogin.html')


def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login Attempt:", usrid, pswd)

        if (usrid, pswd) in [('admin', 'admin'), ('Admin', 'Admin')]:
            return redirect('AdminHome')
        else:
            messages.warning(request, 'Please check your login details')

    return render(request, 'AdminLogin.html')

def AdminHome(request):
    total_users = UserRegistrationModel.objects.count()
    active_users = UserRegistrationModel.objects.filter(status='activated').count()
    pending_approvals = UserRegistrationModel.objects.filter(status='waiting').count()

    context = {
        'total_users': total_users,
        'active_users': active_users,
        'pending_approvals': pending_approvals,
    }
    return render(request, 'admins/AdminHome.html', context)


def ViewRegisteredUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminActivaUsers(request):
    if request.method == 'GET':
        uid = request.GET.get('uid')
        UserRegistrationModel.objects.filter(id=uid).update(status='activated')
        messages.success(request, 'User activated successfully')
    return redirect('ViewRegisteredUsers')


def DeleteUsers(request):
    if request.method == 'GET':
        uid = request.GET.get('uid')
        UserRegistrationModel.objects.filter(id=uid).delete()
        messages.success(request, '✅ User deleted successfully!')
    return redirect('ViewRegisteredUsers')