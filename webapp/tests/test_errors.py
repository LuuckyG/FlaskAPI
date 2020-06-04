import os
import unittest
from unittest import mock
from unittest.mock import patch


from flask import url_for

from webapp.users.models import User
from webapp.tests.test_basic import BasicTest


class TestCustomErrorViews(BasicTest):
    """Unittest case for all error related functions and views.
    This includes restricting user from views such as the admin panel."""

    # Helper functions
    def login(self, client, username, password):
        return client.post(
            url_for('users.login'),
            data=dict(username=username, password=password),
            follow_redirects=True
        )
    
    def add_test_user(self):
        user = User(username='test_user', 
                    email='test_user@test.com')
        user.set_password('password')

        self.db.session.add(user)
        self.db.session.commit()
    
    def add_admin_user(self):
        admin = User(username='admin', 
                    email='admin@test.com',
                    access=3)
        admin.set_password('password')

        self.db.session.add(admin)
        self.db.session.commit()


    # Tests    
    def test_404(self):
        with self.app.test_client() as client:
            path = '/non_existent_endpoint'
            response = client.get(path)
            self.assertEqual(response.status_code, 404)
            self.assertIn(b'Oops. Page Not Found (404)', response.data)
            self.assertIn(b'That page does not exist. Please try a different location.', response.data)


if __name__ == '__main__':
    unittest.main()