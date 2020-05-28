import os
import pytest
import unittest

from webapp import create_app, db, mail
from webapp.config import TestingConfig


class BasicTest(unittest.TestCase):
 
    ### Setup and teardown ###
 
    # Executed prior to each test
    def setUp(self):
        self.app = create_app(TestingConfig)
        print(self.app.config['SECRET_KEY'])
        self.app_context = self.app.app_context()
        self.app_context.push()
        # self.client = self.app.test_client()

        db.create_all()
 
        # Disable sending emails during unit testing
        mail.init_app(self.app)
        self.assertEqual(self.app.debug, False)
 
    # Executed after each test
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    # Tests
    def test_main_page(self):
        with self.app.test_client() as client:
            response = client.get('/', follow_redirects=True)
            self.assertEqual(response.status_code, 200)
 
 
if __name__ == "__main__":
    unittest.main()
