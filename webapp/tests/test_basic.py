import os
import unittest

from webapp import create_app, db, mail
from webapp.config import TestingConfig


class BasicTest(unittest.TestCase):
    
    def setUp(self):
        self.app = create_app(config_class=TestingConfig)
        self.app_context = self.app.app_context()
        self.app_context.push()
         
        self.db = db
        self.db.create_all()
 
        # Disable sending emails during unit testing
        mail.init_app(self.app)
        self.assertEqual(self.app.debug, False)
 
    def tearDown(self):
        self.db.session.remove()
        self.db.drop_all()
        self.app_context.pop()

if __name__ == "__main__":
    unittest.main(verbosity=2)
