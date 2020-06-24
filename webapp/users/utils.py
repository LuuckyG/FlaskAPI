from webapp import mail
from flask_mail import Message
from flask import url_for, current_app, render_template

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager


def send_reset_email(user):
    token = user.get_reset_token()
    msg = Message('Password Reset Request', 
                  sender=current_app.config['ADMINS'][0], 
                  recipients=[user.email])
    
    msg.body = f""" To reset your password, visit the following link:
{url_for('users.reset_token', token=token, _external=True)}

If you did not make this request, then simply ignore this email and no changes will be made.
"""
    mail.send(msg)



class ChromeWebDriver:
    """
    Log in into sharepoint, using the email and password of the current session.
    Use the filename, responding to the file clicked, to find file in sharepoint.
    Open this file and return the page of this opened file.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")

    team_site_url = r'https://evolvalor.sharepoint.com/Shared%20Documents?viewid=32a8b673%2Ddb81%2D46cd%2D8dc4%2D852098b147f6&id=%2FShared%20Documents%2FEvolvalor%20TeamDrive'
    
    def __init__(self, email, password):
        url = 'https://login.microsoftonline.com/'
        self.driver = webdriver.Chrome(ChromeDriverManager().install(), chrome_options=self.options)
        self.driver.get(url)
        self.email = email
        self.password = password
        self.login()
        self.driver.get(self.team_site_url)

    def login(self):
        email_box = self.driver.find_element_by_xpath('//*[@id="i0116"]')
        email_box.send_keys(self.email)
        next_button = self.driver.find_element_by_xpath('//*[@id="idSIButton9"]')
        next_button.click()

        # Password
        password_box = self.driver.find_element_by_xpath('//*[@id="i0118"]')
        password_box.send_keys(self.password)
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="i0281"]/div/div[2]/div[1]/div[2]/div[2]/div/div[2]/div/div[3]/div[2]/div/div/div/div'))).click()

        # Login and Don't remember
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="idBtn_Back"]'))).click()
    
    def search(self, filename):
        self.driver.get(self.team_site_url)

        WebDriverWait(self.driver, 10)
        
        search_box = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@id="sbcId"]/form/input')))
        WebDriverWait(self.driver, 2)
        search_box.clear()
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="sbcId"]/form/input'))).click()
        search_box.send_keys(filename + '\n')

        # Results
        result = WebDriverWait(self.driver, 10).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="appRoot"]/div[1]/div[3]/div/div[2]/div[2]/div[2]/div[2]/div[1]/div[1]/div/div/div[1]/div/div[2]/div/div/div/div/div/div/div[2]/div/div/div/div/div/div[1]/div/div/div[2]/div[2]/div/div/span/span/a')))[0]
        self.driver.get(result.get_attribute('href'))
