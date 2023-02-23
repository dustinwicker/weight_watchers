from __future__ import print_function
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import io
import seaborn as sns
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
# Increase number of rows printed out in console
pd.options.display.max_rows = 200
pd.options.display.min_rows = None
import random

# Change and update directory
os.chdir("C:/Users/wickerd/Desktop")
# List files in current directory by last modified time
files = list(filter(os.path.isfile, os.listdir(os.getcwd())))
files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
print(files)

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
def main():
    global service
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

# Rename columns for easier use
df = df.set_index('Date')
df = df.rename(columns={'Weight (lbs)':'lbs', 'Body fat (%)':'bf'})

# Create columns
df['bf_s_w_r'] = df['bf']
df['bf_s_wo_r'] = df['bf']

# + 1 includes end date
round(len(df)/((df.index.max() - df.index.min()).days + 1)*100,3)

# Data contains rows (dates) where bf % reading is not taken, weight is taken
df_dropna_any = df.dropna(how='any')

# Impute missing body fat % values with Linear Regression - 1
model = LinearRegression().fit(df_dropna_any['lbs'].values.reshape((-1, 1)), df_dropna_any['bf'])
model.score(df_dropna_any['lbs'].values.reshape((-1, 1)), df_dropna_any['bf'])
# Model on scatter plot
model.intercept_, model.coef_
df['imputed_or_not'] = 'n'
df.loc[df['bf'].isna(),'imputed_or_not'] = 'bf_only'
df.loc[df['bf'].isna(),'bf'] = model.predict(df.loc[df['bf'].isna(),'lbs'].values.reshape((-1, 1)))

# Impute 2
# Sampling with replacement
s_w_r = [] 
for i in range(len(df.loc[df.imputed_or_not == 'bf_only'])):
    s_w_r.extend(random.sample(list(df_dropna_any['bf']), 1))

# Impute 3
# Sampling without replacement
s_wo_r = random.sample(list(df_dropna_any['bf']), 30)

df.bf_s_w_r.fillna(pd.Series(s_w_r))
df.bf_s_wo_r.fillna(pd.Series(s_wo_r))


# Put in missing date values (where weight and bf were not recorded) - due to traveling, not having weight scale/bf reader
df = df.reindex(pd.date_range(df.index.min(), df.index.max()))
df.loc[(df['bf'].isna()) & (df['lbs'].isna()),'imputed_or_not'] = 'both'
# Interpolate both weight and bf
df = df.interpolate(method='linear')
len(df.loc[df.imputed_or_not=='n'])/len(df)
# Round to keep 1 sig fig (all that is allowable)
df[['lbs', 'bf']] = df[['lbs', 'bf']].round(1)

# Show family (when I go home...)
plt.plot(df['lbs'])
plt.axvline(pd.to_datetime('2022-09-02'),color='r')
plt.axvline(pd.to_datetime('2022-12-14'),color='r')
plt.axvline(pd.to_datetime('2023-01-01'),color='r')
plt.show()

# Show data imputation (give strategy/reason for imputation) on weight and body fat (indivudally and togther)
# Show effects of data imputation (give visual plot and correlations, statistical tests, normality tests)

# Pearson's correlation
stats.pearsonr(df.loc[df['imputed_or_not'].isin(['n','bf only']),'lbs'], df.loc[df['imputed_or_not'].isin(['n','bf only']),'bf'])
df_pr = stats.pearsonr(df['lbs'], df['bf'])
df_pr

# Poly fit - quadratic bend in correlation plot?
df1 = np.polyfit(df['lbs'], df['bf'],1)#,full=True)
df2 = np.polyfit(df['lbs'], df['bf'],2)#,full=True)
df['df1'] = np.polyval(df1,df['lbs'])
df['df2'] = np.polyval(df2,df['lbs'])
plt.plot(df)
plt.show()

# lbs
df.loc[df['imputed_or_not'].isin(['n','bf only']),'lbs'].describe()
df['lbs'].describe()

fig, (ax1, ax2) = plt.subplots(2,2)
sns.lineplot(data=df.loc[df['imputed_or_not'].isin(['n','bf only']), 'lbs'],ax=ax1[0])
sns.histplot(data=df.loc[df['imputed_or_not'].isin(['n','bf only']), 'lbs'],ax=ax1[1])
sns.lineplot(data=df['lbs'],ax=ax2[0])
sns.histplot(data=df['lbs'],ax=ax2[1])
#sns.lineplot(data=df['lbs'],ax=ax2, marker='o')
#sns.lineplot(data=df['lbs'],hue='imputed_or_not',ax=ax2)
plt.show()

# bf
df.loc[df['imputed_or_not'].isin(['n']),'bf'].describe()
df['bf'].describe()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,2)
sns.lineplot(data=df.loc[df['imputed_or_not'].isin(['n']),'bf'],ax=ax1[0])
sns.histplot(data=df.loc[df['imputed_or_not'].isin(['n']),'bf'],ax=ax1[1])
sns.lineplot(data=df['bf'],ax=ax2[0])
sns.histplot(data=df['bf'],ax=ax2[1])
sns.lineplot(data=df['bf_s_w_r'],ax=ax3[0])
sns.histplot(data=df['bf_s_w_r'],ax=ax3[1])
sns.lineplot(data=df['bf_s_wo_r'],ax=ax4[0])
sns.histplot(data=df['bf_s_wo_r'],ax=ax4[1])
plt.show()


sns.scatterplot(df['lbs'], df['bf'], hue = df['imputed_or_not'])
plt.scatter(df_dropna_any['lbs'], df_dropna_any['bf'])
plt.show()


