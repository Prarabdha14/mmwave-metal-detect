import requests
resp = requests.post(
    "http://127.0.0.1:5001/infer",
    json={"simulate":[{"r":6,"v":0.5,"rcs":3,"snr_db":22}]}
)
print(resp.json())
