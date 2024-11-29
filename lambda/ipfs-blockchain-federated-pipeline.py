import json
import requests
import base64
import io
from io import BytesIO

def upload_model_to_ipfs(index, model_base64):
    # Save the model to a buffer
    #uffer = BytesIO()
    #torch.save(model, buffer)
    #buffer.seek(0)
    
    # Encode the model to Base64
    #model_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    #print(model_base64)
    
    # Upload the model to IPFS
    url = "https://2zk0vq0ll6.execute-api.us-east-1.amazonaws.com/default/ipfs-file-streaming"
    payload = {
        "action": "0",
        "fileName": f"aggregated_model{index}.pth",
        "fileContent": model_base64
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(f"Model uploaded to IPFS with CID: {response.json()['cid']}")
    else:
        raise Exception(f"Failed to upload model to IPFS. Status code: {response.status_code}, Response: {response.content}")
    
    return response.json()['cid']

def upload_hash_to_blockchain(index, hash):
    url = "https://82o2i4mwfc.execute-api.us-east-1.amazonaws.com/default/blockchain-storage"
    payload = {
        "action": "0",
        "clientIndex": index,
        "content": hash
    }
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(f"Hash uploaded to blockchain for client {index}")
    else:
        raise Exception(f"Failed to upload hash to blockchain. Status code: {response.status_code}, Response: {response.content}")

#A lambda function to handle whole pipeline
#Read in model, upload to IPFS, invoke blockchain, invoke 
def lambda_handler(event, context):
    #Get encoded model, store on IPFS
    #Store hash on blockchain(add flag)
    #Call federated learning

    body = event.get('body', '{}') 
    parsed_body = json.loads(body)
    
    #clientIndex = the client updated the model
    updatedClientIndex = parsed_body.get('updatedClientIndex')
    blockchainEnabled = parsed_body.get('blockchainEnabled')
    modelEncoded = parsed_body.get('modelEncoded')

    newHash = upload_model_to_ipfs(updatedClientIndex, modelEncoded)

    if blockchainEnabled == "1":
        upload_hash_to_blockchain(updatedClientIndex, newHash)

    url = "https://federated-learning-473375422539.us-central1.run.app"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "updatedClientIndex": updatedClientIndex,
        "blockchainEnabled": blockchainEnabled
    }

    try:
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            response_data = response.json()
            return {
                "statusCode": 200,
                "body": json.dumps(response_data)
            }
        else:
            return {
                "statusCode": response.status_code,
                "body": f"Error: {response.text}"
            }
    except Exception as e:
        # Handle exceptions (e.g., connection issues)
        return {
            "statusCode": 500,
            "body": f"An error occurred: {str(e)}"
        }

'''
url = "https://g6jentj8ia.execute-api.us-east-1.amazonaws.com/default/ipfs-blockchain-federated-pipeline"
payload = {
    "updatedClientIndex": "0",
    "blockchainEnabled": "0",
    "modelEncoded": "UEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAQABIAYXJjaGl2ZS9kYXRhLnBrbEZCDgBaWlpaWlpaWlpaWlpaWoACY2NvbGxlY3Rpb25zCk9yZGVyZWREaWN0CnEAKVJxAShYCQAAAGZjLndlaWdodHECY3RvcmNoLl91dGlscwpfcmVidWlsZF90ZW5zb3JfdjIKcQMoKFgHAAAAc3RvcmFnZXEEY3RvcmNoCkZsb2F0U3RvcmFnZQpxBVgBAAAAMHEGWAMAAABjcHVxB0sKdHEIUUsASwFLCoZxCUsKSwGGcQqJaAApUnELdHEMUnENWAcAAABmYy5iaWFzcQ5oAygoaARoBVgBAAAAMXEPaAdLAXRxEFFLAEsBhXERSwGFcRKJaAApUnETdHEUUnEVdX1xFlgJAAAAX21ldGFkYXRhcRdoAClScRgoWAAAAABxGX1xGlgHAAAAdmVyc2lvbnEbSwFzWAIAAABmY3EcfXEdaBtLAXN1c2IuUEsHCLgNEJ06AQAAOgEAAFBLAwQAAAgIAAAAAAAAAAAAAAAAAAAAAAAADgAKAGFyY2hpdmUvZGF0YS8wRkIGAFpaWlpaWtEIPr39I7E9tDm1vToOlj4x3He+sACKPaxmjj6Lkou+wN0Rvl0XMT5QSwcI9LI12SgAAAAoAAAAUEsDBAAACAgAAAAAAAAAAAAAAAAAAAAAAAAOABwAYXJjaGl2ZS9kYXRhLzFGQhgAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpai2SDPVBLBwhDCgQcBAAAAAQAAABQSwMEAAAICAAAAAAAAAAAAAAAAAAAAAAAAA8APwBhcmNoaXZlL3ZlcnNpb25GQjsAWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlpaWlozClBLBwjRnmdVAgAAAAIAAABQSwECAAAAAAgIAAAAAAAAuA0QnToBAAA6AQAAEAAAAAAAAAAAAAAAAAAAAAAAYXJjaGl2ZS9kYXRhLnBrbFBLAQIAAAAACAgAAAAAAAD0sjXZKAAAACgAAAAOAAAAAAAAAAAAAAAAAIoBAABhcmNoaXZlL2RhdGEvMFBLAQIAAAAACAgAAAAAAABDCgQcBAAAAAQAAAAOAAAAAAAAAAAAAAAAAPgBAABhcmNoaXZlL2RhdGEvMVBLAQIAAAAACAgAAAAAAADRnmdVAgAAAAIAAAAPAAAAAAAAAAAAAAAAAFQCAABhcmNoaXZlL3ZlcnNpb25QSwYGLAAAAAAAAAAeAy0AAAAAAAAAAAAEAAAAAAAAAAQAAAAAAAAA8wAAAAAAAADSAgAAAAAAAFBLBgcAAAAAxQMAAAAAAAABAAAAUEsFBgAAAAAEAAQA8wAAANICAAAAAA=="
}
response = requests.post(url, json=payload)

'''