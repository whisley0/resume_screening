## resume_screening

- POC Deploymnet of the app on heroku
- This app use nlp matcher to match keyword from CV and class them into category, visualize the result into graph
- use matcher to predict the name of the CV owner and perform a search of Linked URL online, login to linkedin and scrape the owner's interest page

some note on environment setting for chromedriver on heroku:

# 1.Heroku Buildpacks🔗
heroku/python

https://github.com/heroku/heroku-buildpack-google-chrome

https://github.com/heroku/heroku-buildpack-chromedriver

# 2. Heroku environmental variables
CHROMEDRIVER_PATH	/app/.chromedriver/bin/chromedriver
GOOGLE_CHROME_BIN	/app/.apt/usr/bin/google-chrome

# 3. how to call the path
driver = webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"), chrome_options=chrome_options)

