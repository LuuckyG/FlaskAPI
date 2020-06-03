# import os
# import unittest

# from webapp.tests.test_basic import BasicTest


# class BlueprintOrAppTestCase(BasicTest):

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
#     unittest.main()