from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager


class ChromeWebDriver:
    """
    Log in into sharepoint, using the email and password of the current session.
    Use the filename, responding to the file clicked, to find file in sharepoint.
    Open this file and return the page of this opened file.
    """
    options = webdriver.ChromeOptions()
    options.binary_location = os.environ.get("GOOGLE_CHROME_BIN")
    options.add_argument("--headless")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    
    url = 'https://login.microsoftonline.com/'
    team_site_url = r'https://evolvalor.sharepoint.com/Shared%20Documents?viewid=32a8b673%2Ddb81%2D46cd%2D8dc4%2D852098b147f6&id=%2FShared%20Documents%2FEvolvalor%20TeamDrive'
    
    def __init__(self, email, password):
        self.driver = webdriver.Chrome(executable_path=str(os.environ.get("CHROMEDRIVER_PATH")), chrome_options=self.options)
        self.driver.get(self.url)
        self.email = email
        self.password = password
        self.login()
        self.driver.get(self.team_site_url)

    def login(self):
        """Log in to sharepoint"""
        # Email
        email_box = self.driver.find_element_by_xpath('//*[@id="i0116"]')
        email_box.send_keys(self.email)
        next_button = WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="idSIButton9"]')))
        next_button.click()

        # Password
        password_box = self.driver.find_element_by_xpath('//*[@id="i0118"]')
        password_box.send_keys(self.password)
        WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="i0281"]/div/div[2]/div[1]/div[2]/div[2]/div/div[2]/div/div[3]/div[2]/div/div/div/div'))).click()

        # Login and don't stay logged in
        WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="idBtn_Back"]'))).click()
    
    def search(self, filename):
        """Search file `filename` from the teamsite main page."""
        # Back to team drive main page
        if self.driver.current_url != self.team_site_url:
            self.driver.get(self.team_site_url)

        WebDriverWait(self.driver, 10)
        
        # Search
        search_box = WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.XPATH, '//*[@id="sbcId"]/form/input')))
        WebDriverWait(self.driver, 5)
        search_box.clear()
        WebDriverWait(self.driver, 30).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="sbcId"]/form/input'))).click()
        search_box.send_keys(filename + '\n')

        # Results
        result = WebDriverWait(self.driver, 30).until(EC.presence_of_all_elements_located((By.XPATH, '//*[@id="appRoot"]/div[1]/div[3]/div/div[2]/div[2]/div[2]/div[2]/div[1]/div[1]/div/div/div[1]/div/div[2]/div/div/div/div/div/div/div[2]/div/div/div/div/div/div[1]/div/div/div[2]/div[2]/div/div/span/span/a')))[0]
        self.driver.get(result.get_attribute('href'))

        
FIELDS_OF_INTEREST = ['key_terms', 
                      'title', 
                      'aanleiding',
                      'opl',
                      't_knel',
                      'prog',
                      't_nieuw']


def combine_search_form_inputs(inputs):
    search_query = ''

    for field in FIELDS_OF_INTEREST:
        try:
            value = inputs[field]
            if not value == '':

                if not isinstance(value, str):
                    value = str(value)

                search_query += value + '.\n'

        except KeyError:
            continue

    return search_query
