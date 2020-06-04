import os

from flask import url_for, request
from flask_login import current_user

from webapp.tests.test_basic import BasicTest
from webapp.users.models import User, Task


class TestUserViews(BasicTest):
    """Unittest case for all user related functions and views."""

    # Helper functions
    def register(self, client, username, email, password, confirm_password, access=1):
        return client.post(
            url_for('users.register'),
            data=dict(username=username, email=email, password=password, confirm_password=confirm_password),
            follow_redirects=True
        )

    def login(self, client, username, password):
        return client.post(
            url_for('users.login'),
            data=dict(username=username, password=password),
            follow_redirects=True
        )
    
    def logout(self, client):
        return client.get(
            url_for('users.logout'),
            follow_redirects=True
        )

    def add_test_user(self):
        user = User(username='test_user', 
                    email='test_user@test.com')
        user.set_password('password')

        self.db.session.add(user)
        self.db.session.commit()

    # Tests
    def test_empty_database(self):
        users = User.query.all()
        tasks = Task.query.all()

        self.assertEqual(users, [])
        self.assertEqual(tasks, [])


    def test_password_hashing(self):
        test_user = User(username='test_user', 
					 email='test_user@test.com')
        test_user.set_password('password')

        self.assertFalse(test_user.check_password('test'))
        self.assertTrue(test_user.check_password('password'))

    
    def test_register(self):
        with self.app.test_client() as client:
            self.assertEqual(client.get("/register").status_code, 200)

            response = self.register(client, 'test_user', 'test_user@test.com', 'password', 'password')
            response2 = self.register(client, 'second_user', 'second_user@test.com', 'password', 'password')

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response2.status_code, 200)

            self.assertIn(b'Account created for test_user', response.data)
            self.assertIn(b'Account created for second_user', response2.data)

            users = User.query.all()
            self.assertEqual(len(users), 2)
            self.assertEqual(users[0].username, 'test_user')
            self.assertEqual(users[1].username, 'second_user')
      

    def test_invalid_user_registration_different_passwords(self):
        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                response = self.register(client, 'test_user', 'test_user@test.com', 'password', 'password2')
                self.assertIn(b'Field must be equal to password.', response.data)


    def test_invalid_user_registration_existing_username(self):
        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                response = self.register(client, 'test_user', 'test_user@test.com', 'password', 'password')
                response2 = self.register(client, 'test_user', 'second_user@test.com', 'password', 'password')
                
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response2.status_code, 200)
                self.assertIn(b'That username is taken. Please choose a different one.', response2.data)
    

    def test_invalid_user_registration_existing_email(self):
        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                response = self.register(client, 'test_user', 'test_user@test.com', 'password', 'password')
                response2 = self.register(client, 'second_user', 'test_user@test.com', 'password', 'password')
                
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response2.status_code, 200)
                self.assertIn(b'That email is taken. Please choose a different one.', response2.data)
    

    def test_login(self):
        self.add_test_user()

        with self.app.test_client() as client:
            self.assertEqual(client.get("/login").status_code, 200)

            response = self.login(client, username='test_user', password='password')
            
            self.assertEqual(response.status_code, 200)
            self.assertTrue(request.path == url_for('main.index'))
            self.assertTrue(current_user.username == 'test_user')
            self.assertFalse(current_user.is_anonymous)
    

    def test_incorrect_login(self):
        self.add_test_user()

        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                self.assertEqual(client.get("/login").status_code, 200)

                response = self.login(client, username='wrong_username', password='password')
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'Login Unsuccesful. Please check username and password', response.data)
                self.assertTrue(current_user.is_anonymous)
                self.assertTrue(request.path == url_for('users.login'))

                response = self.login(client, username='test_user', password='wrong_password')
                self.assertEqual(response.status_code, 200)
                self.assertIn(b'Login Unsuccesful. Please check username and password', response.data)
                self.assertTrue(current_user.is_anonymous)
                self.assertTrue(request.path == url_for('users.login'))
            

    def test_logout(self):
        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                self.assertEqual(self.logout(client).status_code, 200)


    def test_login_logout(self):
        self.add_test_user()

        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                response = self.login(client, username='test_user', password='password')    
                self.assertTrue(current_user.username == 'test_user')

                self.logout(client)
                self.assertTrue(current_user.is_anonymous)


    def test_redirection_authenticated_user(self):
        self.add_test_user()

        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                response = self.login(client, username='test_user', password='password')

                client.get(url_for('users.login'), follow_redirects=True)
                self.assertTrue(request.path == url_for('main.index'))

                client.get(url_for('users.register'), follow_redirects=True)
                self.assertTrue(request.path == url_for('main.index'))

                self.logout(client)
                client.get(url_for('users.login'), follow_redirects=True)
                self.assertTrue(request.path == url_for('users.login'))

                client.get(url_for('users.register'), follow_redirects=True)
                self.assertTrue(request.path == url_for('users.register'))
        
    
    def test_redirection(self):
        self.add_test_user()
        
        with self.app_context, self.app.test_request_context():
            with self.app.test_client() as client:
                self.login(client, username='test_user', password='password')

                client.get(url_for('users.history'))
                self.assertEqual(request.path, url_for('users.history'))

                client.get(url_for('users.account'))
                self.assertEqual(request.path, url_for('users.account'))               


if __name__ == "__main__":
    unittest.main(verbosity=2)
