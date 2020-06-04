from flask import (Blueprint, current_app, redirect, abort,
                    render_template, url_for, request, flash, session)
from flask_login import login_user, current_user, logout_user, login_required

from webapp.utils import requires_access_level
from webapp.users.models import ACCESS


admins = Blueprint('admins', __name__)


@admins.route('/admin', methods=['GET', 'POST'])
@login_required
def index(): 
    return render_template('admin/index.html')
