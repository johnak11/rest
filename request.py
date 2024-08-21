import requests

url = 'http://localhost:5000//predict_api'
r = requests.post(url,json={'R&D Spend':122, 'Administration':333, 'Marketing Spend':444,'State':1})

print(r.json())