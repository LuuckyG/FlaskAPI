from flask import (Blueprint, current_app, redirect, 
                    render_template, url_for, request, flash, session)
from flask_login import login_user, current_user, logout_user, login_required


admins = Blueprint('admins', __name__)


@admins.route('/admin', methods=['GET', 'POST'])
@login_required
def index(): 
    if current_user.is_authenticated:
        return render_template('admin/index.html')
    return redirect(url_for('users.login'))
