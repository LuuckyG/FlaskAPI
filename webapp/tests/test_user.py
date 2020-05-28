import os

from webapp import create_app, db, mail
from webapp.tests.test_basic import BasicTest
from webapp.config import TestingConfig


class TestUser(BasicTest):
    """Unittest case for all user related functions."""

    def register(self, email, password, confirm):
        return self.app.post(
            '/register',
            data=dict(email=email, password=password, confirm=confirm),
            follow_redirects=True
        )
    
    def login(self, email, password):
        return self.app.post(
            '/login',
            data=dict(email=email, password=password),
            follow_redirects=True
        )
    
    def logout(self):
        return self.app.get(
            '/logout',
            follow_redirects=True
        )
    
    def test_login_logout(client):
        """Make sure login and logout works."""

        with self.app.test_client() as client:
            rv = login(client, webapp.app.config['USERNAME'], webapp.app.config['PASSWORD'])
            assert b'You were logged in' in rv.data

            rv = logout(client)
            assert b'You were logged out' in rv.data

            rv = login(client, webapp.app.config['USERNAME'] + 'x', webapp.app.config['PASSWORD'])
            assert b'Invalid username' in rv.data

            rv = login(client, webapp.app.config['USERNAME'], webapp.app.config['PASSWORD'] + 'x')
            assert b'Invalid password' in rv.data
 
if __name__ == "__main__":
    unittest.main()



# import os
# import unittest
# from webapp.config import TestingConfig


# class UserModelCase(unittest.TestCase):
#     def setUp(self):
#         self.app = create_app(TestingConfig)
#         self.app_context = self.app.app_context()
#         self.app_context.push()
#         db.create_all()

#     def tearDown(self):
#         db.session.remove()
#         db.drop_all()
#         self.app_context.pop()


# class BlueprintOrAppTestCase(unittest.TestCase):

#     def setUp(self):
#         self.client = app.test_client()

#     def test_200(self):
#         resp = self.client.get('/blue/hello')
#         self.assertEqual(resp.status_code, 200)
#         self.assertEqual(resp.get_data(True), 'hello world!')

#     def test_404_main(self):
#         with app.test_client() as client:
#             resp = client.get('/notExist')
#             self.assertEqual(resp.status_code, 404)
#             self.assertEqual(resp.get_data(True), 'app 404')

#     def test_404_blueprint(self):
#         with app.test_client() as client:
#             resp = client.get('/blue/notExist')
#             self.assertEqual(resp.status_code, 404)
#             self.assertEqual(resp.get_data(True), 'myblueprint 404')
#             self.assertEqual(request.blueprint, 'myblueprint')

#     def test_404_forced_blueprint(self):
#         with app.test_client() as client:
#             resp = client.get('/blue/forced_404')
#             self.assertEqual(resp.status_code, 404)
#             self.assertEqual(resp.get_data(True), 'myblueprint 404')
#             self.assertEqual(request.blueprint, 'myblueprint')


# if __name__ == '__main__':
#     # app.run(host="0.0.0.0", use_reloader=True)
#     unittest.main()
