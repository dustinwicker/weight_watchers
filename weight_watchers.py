import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
# Increase number of rows printed out in console
pd.options.display.max_rows = 200
pd.options.display.min_rows = None

# Change and update directory
os.chdir("C:/Users/wickerd/Desktop")
# List files in current directory by last modified time
files = list(filter(os.path.isfile, os.listdir(os.getcwd())))
files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
print(files)





from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)

        # Call the Drive v3 API
        results = service.files().list(
            pageSize=10, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
            return
        print('Files:')
        for item in items:
            print(u'{0} ({1})'.format(item['name'], item['id']))
    except HttpError as error:
        # TODO(developer) - Handle errors from drive API.
        print(f'An error occurred: {error}')


if __name__ == '__main__':
    main()

results = service.files().list(pageSize=1, fields="files(modifiedTime,name,id)", orderBy="modifiedTime desc", q="'" + '1oAwcYkyRKKeTM8wgZDxBJwcTFd2xQm0zy73pjbXZYmk' + "' in parents and mimeType = 'application/vnd.google-apps.spreadsheet'", supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
items = service.files().list(pageSize=1).execute().get('files', [])
if items:
    file_id = items[0]['id']
    file_name = items[0]['name']
    request = service.files().export_media(fileId=file_id, mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    fh = io.FileIO(file_name + '.xlsx', mode='wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print('Download %d%%.' % int(status.progress() * 100))
    df = pd.read_excel('Daily Weight Measurements\xa0.xlsx')

# + 1 includes end date
round(len(df)/((df.iloc[-1]['Date'] - df.iloc[0]['Date']).days + 1)*100,3)

from sklearn.linear_model import LinearRegression
df_dropna_any = df.dropna(how='any')


plt.plot(df_dropna_any[['Body fat (%)', 'Weight (lbs)']])
plt.scatter(df_dropna_any['Weight (lbs)'], df_dropna_any['Body fat (%)'])
plt.show()

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(df_dropna_any['Weight (lbs)'].values.reshape((-1, 1)), df_dropna_any['Body fat (%)'])
model.score(df_dropna_any['Weight (lbs)'].values.reshape((-1, 1)), df_dropna_any['Body fat (%)'])
model.intercept_
model.coef_
df['Body fat (%)m2'] = df['Body fat (%)']
df['imputed_or_not'] = 'n'
df.loc[df['Body fat (%)'].isna(),'imputed_or_not'] = 'y'
df.loc[df['Body fat (%)'].isna(),'Body fat (%)'] = model.predict(df.loc[df['Body fat (%)'].isna(),'Weight (lbs)'].values.reshape((-1, 1)))
df

model = LinearRegression().fit(df_dropna_any['Body fat (%)'].values.reshape((-1, 1)), df_dropna_any['Weight (lbs)'])
model.score(df_dropna_any['Body fat (%)'].values.reshape((-1, 1)), df_dropna_any['Weight (lbs)'])
model.intercept_
model.coef_

df.loc[df['imputed_or_not']=='y','Body fat (%)m2'] = model.predict(df.loc[df['imputed_or_not']=='y','Weight (lbs)'].values.reshape((-1, 1)))
df

df.loc[df['Body fat (%)'].isna()]

plt.plot(df['Weight (lbs)'])
plt.show()

plt.plot(df['Body fat (%)'])
plt.show()

df_pr = stats.pearsonr(df['Weight (lbs)'], df['Body fat (%)'])
df_pr
df1 = np.polyfit(df['Weight (lbs)'], df['Body fat (%)'],1)#,full=True)
df2 = np.polyfit(df['Weight (lbs)'], df['Body fat (%)'],2)#,full=True)
df['df1'] = np.polyval(df1,df['Weight (lbs)'])
df['df2'] = np.polyval(df2,df['Weight (lbs)'])
plt.plot(df)
plt.show()

plt.hist(df['Weight (lbs)'])
plt.show()
df.columns

plt.hist(df['Body fat (%)'])
plt.show()

df = df.set_index('Date')
import seaborn as sns
plt.scatter(df['Weight (lbs)'], df['Body fat (%)'])
sns.scatterplot(df['Weight (lbs)'], df['Body fat (%)'], hue = df['imputed_or_not'])
plt.plot(df['df2'])
plt.show()


pd.corr(df['Weight (lbs)'], df['Body fat (%)'])

df['Weight (lbs)'].describe()
df['Body fat (%)'].describe()


