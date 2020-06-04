import os
import unittest

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
    def test_403(self):
        self.add_test_user()

        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:

                response = self.login(client, username='test_user', password='password')
                self.assertEqual(response.status_code, 200)

                response = client.get('/admin')
                self.assertEqual(response.status_code, 403)
                self.assertIn(b"You don't have permission to do that. (403)", response.data)
                self.assertIn(b'Please check your account and try again.', response.data)
    

    def test_admin_permission(self):
        self.add_admin_user()

        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:

                response = self.login(client, username='admin', password='password')
                self.assertEqual(response.status_code, 200)

                response = client.get('/admin')
                self.assertEqual(response.status_code, 200)

    
    def test_404(self):
        with self.app.test_client() as client:
            path = '/non_existent_endpoint'
            response = client.get(path)
            self.assertEqual(response.status_code, 404)
            self.assertIn(b'Oops. Page Not Found (404)', response.data)
            self.assertIn(b'That page does not exist. Please try a different location.', response.data)


if __name__ == '__main__':
    unittest.main()